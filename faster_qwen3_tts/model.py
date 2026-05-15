"""
FasterQwen3TTS: Real-time TTS using CUDA graph capture.

Wrapper class that provides a Qwen3-TTS API while using
CUDA graphs for 6-10x speedup.
"""
import logging
import os
from collections import OrderedDict
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from .utils import suppress_flash_attn_warning

logger = logging.getLogger(__name__)




class FasterQwen3TTS:
    """
    Qwen3-TTS model with CUDA graphs for real-time inference.
    
    Compatible API with Qwen3TTSModel, but uses CUDA graph
    capture for 6-10x speedup on NVIDIA GPUs.
    """
    
    def __init__(
        self,
        base_model,
        predictor_graph,
        talker_graph,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        audio_dtype: Optional[torch.dtype] = None,
        max_seq_len: int = 2048,
    ):
        self.model = base_model  # The qwen-tts Qwen3TTSModel instance
        self.predictor_graph = predictor_graph
        self.talker_graph = talker_graph
        self.device = device
        self.dtype = dtype
        self.audio_dtype = audio_dtype or dtype
        self.max_seq_len = max_seq_len
        self.sample_rate = self._infer_sample_rate(base_model)
        self._warmed_up = False
        self.max_voice_prompt_cache_entries = max(
            0,
            int(os.environ.get("QWEN_TTS_VOICE_PROMPT_CACHE_ENTRIES", "8")),
        )
        self._voice_prompt_cache = OrderedDict()  # (ref_audio, ref_text, mode) -> (vcp, ref_ids)

    def _get_voice_prompt_cache(self, cache_key):
        if cache_key not in self._voice_prompt_cache:
            return None
        self._voice_prompt_cache.move_to_end(cache_key)
        return self._voice_prompt_cache[cache_key]

    def _set_voice_prompt_cache(self, cache_key, value) -> None:
        if self.max_voice_prompt_cache_entries <= 0:
            return
        self._voice_prompt_cache[cache_key] = value
        self._voice_prompt_cache.move_to_end(cache_key)
        while len(self._voice_prompt_cache) > self.max_voice_prompt_cache_entries:
            self._voice_prompt_cache.popitem(last=False)

    @staticmethod
    def _get_speech_tokenizer(base_model):
        """Return the nested qwen-tts speech tokenizer when available."""
        return getattr(getattr(base_model, "model", None), "speech_tokenizer", None)

    @property
    def speech_tokenizer(self):
        """Expose the codec decoder on the wrapper's public surface."""
        speech_tokenizer = self._get_speech_tokenizer(self.model)
        if speech_tokenizer is None:
            raise AttributeError("Underlying model does not expose a speech_tokenizer")
        return speech_tokenizer

    @staticmethod
    def _infer_sample_rate(base_model) -> int:
        """Infer output audio sample rate from qwen-tts internals."""
        # Qwen3-TTS model IDs include "12Hz", but that is codec frame-rate (tokens/s),
        # not waveform sampling rate. Generated audio is 24kHz.
        sample_rate = None

        speech_tokenizer = FasterQwen3TTS._get_speech_tokenizer(base_model)
        if speech_tokenizer is not None:
            sample_rate = getattr(speech_tokenizer, "sample_rate", None)

        if sample_rate is None:
            sample_rate = getattr(base_model, "sample_rate", None)

        if sample_rate is None:
            logger.warning(
                "Could not infer sample rate from base model; defaulting to 24000 Hz."
            )
            return 24000

        return int(sample_rate)

    @staticmethod
    def _parse_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
        """Parse common dtype spellings."""
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype
        normalized = dtype.lower().replace("torch.", "")
        aliases = {
            "auto": None,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float": torch.float32,
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported dtype {dtype!r}. Expected auto, bf16, fp16, or fp32."
            )
        return aliases[normalized]

    @staticmethod
    def _auto_generation_dtype(device: str) -> torch.dtype:
        """Pick the stable generation dtype for the current device."""
        if device.startswith("cuda") and torch.cuda.is_available():
            device_index = torch.device(device).index
            device_index = device_index if device_index is not None else torch.cuda.current_device()
            major, _minor = torch.cuda.get_device_capability(device_index)
            if major < 8:
                return torch.float32
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        return torch.float32 if device == "cpu" else torch.bfloat16

    @staticmethod
    def _resolve_dtypes(
        dtype: Optional[Union[str, torch.dtype]],
        audio_dtype: Optional[Union[str, torch.dtype]],
        device: str,
    ) -> Tuple[torch.dtype, torch.dtype]:
        generation_dtype = FasterQwen3TTS._parse_dtype(dtype)
        if generation_dtype is None:
            generation_dtype = FasterQwen3TTS._auto_generation_dtype(device)

        if isinstance(audio_dtype, str) and audio_dtype.lower() == "same":
            resolved_audio_dtype = generation_dtype
        else:
            resolved_audio_dtype = FasterQwen3TTS._parse_dtype(audio_dtype)
            if resolved_audio_dtype is None:
                # Keep audio-side modules fp32 when the caller explicitly experiments
                # with fp16 generation. Full fp16 can make the codec decoder noisy.
                resolved_audio_dtype = (
                    torch.float32 if generation_dtype is torch.float16 else generation_dtype
                )
        return generation_dtype, resolved_audio_dtype

    @staticmethod
    def _is_pre_ampere_cuda(device: str) -> bool:
        if not device.startswith("cuda") or not torch.cuda.is_available():
            return False
        device_index = torch.device(device).index
        device_index = device_index if device_index is not None else torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(device_index)
        return major < 8

    @staticmethod
    def _resolve_attn_implementation(attn_implementation: str, device: str) -> str:
        if attn_implementation.lower() != "auto":
            return attn_implementation
        # On V100/Volta, PyTorch SDPA cannot use flash or cuDNN attention, and
        # the memory-efficient path is slower than eager for this token-by-token
        # decode shape. Ampere+ keeps SDPA as the default.
        return "eager" if FasterQwen3TTS._is_pre_ampere_cuda(device) else "sdpa"

    @staticmethod
    def _apply_linear_precision(base_model, linear_precision: str, device: str, dtype: torch.dtype) -> int:
        """Apply optional inner-fp16 Linear wrappers to stable fp32 models."""
        mode = linear_precision.lower()
        if mode == "auto":
            mode = "fp16_inner" if dtype is torch.float32 and FasterQwen3TTS._is_pre_ampere_cuda(device) else "none"
        if mode in ("none", "off", "disabled"):
            return 0
        if mode not in ("fp16_inner", "talker_fp16_inner"):
            raise ValueError(
                "Unsupported linear_precision "
                f"{linear_precision!r}. Expected auto, none, or fp16_inner."
            )
        if dtype is not torch.float32:
            raise ValueError("linear_precision='fp16_inner' requires generation dtype fp32")

        from .mixed_precision import replace_linear_modules

        talker_model = base_model.model.talker.model
        target_names = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        return replace_linear_modules(
            talker_model,
            lambda name, _linear: name in target_names,
            inner_dtype=torch.float16,
        )

    @staticmethod
    def _apply_layer_precision(base_model, layer_precision: str, device: str, dtype: torch.dtype) -> list[int]:
        """Wrap selected talker decoder layers as fp16 islands."""
        mode = layer_precision.lower()
        if mode == "auto":
            mode = "all" if dtype is torch.float32 and FasterQwen3TTS._is_pre_ampere_cuda(device) else "none"
        if mode in ("none", "off", "disabled"):
            return []
        if dtype is not torch.float32:
            raise ValueError("layer_precision requires generation dtype fp32")

        from .mixed_precision import wrap_decoder_layers

        return wrap_decoder_layers(base_model.model.talker.model, mode)

    @staticmethod
    def _apply_predictor_layer_precision(base_model, predictor_layer_precision: str, device: str, dtype: torch.dtype) -> list[int]:
        """Wrap selected code-predictor decoder layers as fp16 islands."""
        mode = predictor_layer_precision.lower()
        if mode == "auto":
            # The predictor directly emits codec codebooks. Small numerical
            # changes here can become short audible glitches, while the module is
            # much smaller than the main talker stack. Keep it stable by default.
            mode = "none"
        if mode in ("none", "off", "disabled"):
            return []
        if dtype is not torch.float32:
            raise ValueError("predictor_layer_precision requires generation dtype fp32")

        from .mixed_precision import wrap_decoder_layers

        return wrap_decoder_layers(base_model.model.talker.code_predictor.model, mode)

    @staticmethod
    def _apply_audio_decoder_precision(base_model, audio_decoder_precision: str, device: str, dtype: torch.dtype) -> dict[str, object]:
        """Apply fp16 islands to the speech-tokenizer decoder."""
        mode = audio_decoder_precision.lower()
        if mode == "auto":
            # The codec/audio decoder is downstream of token generation and is
            # sensitive to quantizer/synthesis precision. It is also a smaller
            # share of end-to-end latency than the autoregressive transformer
            # loops, so leave it in the stable dtype unless explicitly enabled.
            mode = "none"
        if mode in ("none", "off", "disabled"):
            return {
                "transformer_layers": [],
                "synthesis_wrappers": 0,
                "quantizer": False,
                "projection_wrappers": 0,
            }
        if mode != "fp16":
            raise ValueError(
                f"Unsupported audio_decoder_precision {audio_decoder_precision!r}. Expected auto, none, or fp16."
            )
        if dtype is not torch.float32:
            raise ValueError("audio_decoder_precision='fp16' requires generation dtype fp32")

        speech_tokenizer = FasterQwen3TTS._get_speech_tokenizer(base_model)
        if speech_tokenizer is None or getattr(speech_tokenizer, "model", None) is None:
            return {
                "transformer_layers": [],
                "synthesis_wrappers": 0,
                "quantizer": False,
                "projection_wrappers": 0,
            }

        from .mixed_precision import InnerDtypeLinear, wrap_decoder_layers, wrap_tokenizer_decoder_synthesis

        decoder = speech_tokenizer.model.decoder
        transformer_layers = wrap_decoder_layers(decoder.pre_transformer, "all")
        synthesis_wrappers = wrap_tokenizer_decoder_synthesis(decoder, output_dtype=torch.float32)
        decoder.quantizer.to(dtype=torch.float16)
        projection_wrappers = 0
        decoder.pre_transformer.input_proj = InnerDtypeLinear(
            decoder.pre_transformer.input_proj,
            torch.float16,
        )
        projection_wrappers += 1
        decoder.pre_transformer.output_proj = InnerDtypeLinear(
            decoder.pre_transformer.output_proj,
            torch.float16,
        )
        projection_wrappers += 1
        decoder.pre_transformer.norm.to(dtype=torch.float16)
        return {
            "transformer_layers": transformer_layers,
            "synthesis_wrappers": synthesis_wrappers,
            "quantizer": True,
            "projection_wrappers": projection_wrappers,
        }

    @staticmethod
    def _apply_large_block_precision(base_model, large_block_precision: str, device: str, dtype: torch.dtype) -> int:
        """Move validated large non-decoder blocks to fp16."""
        mode = large_block_precision.lower()
        if mode == "auto":
            # Embeddings, speaker encoder, and final norms are low Tensor-Core
            # value compared with decoder layers and affect voice/tone stability.
            mode = "none"
        if mode in ("none", "off", "disabled"):
            return 0
        if mode != "fp16":
            raise ValueError(
                f"Unsupported large_block_precision {large_block_precision!r}. Expected auto, none, or fp16."
            )
        if dtype is not torch.float32:
            raise ValueError("large_block_precision='fp16' requires generation dtype fp32")

        count = 0
        talker = base_model.model.talker
        talker.model.text_embedding.to(dtype=torch.float16)
        count += 1
        talker.model.codec_embedding.to(dtype=torch.float16)
        count += 1
        talker.code_predictor.model.codec_embedding.to(dtype=torch.float16)
        count += 1

        speech_tokenizer = FasterQwen3TTS._get_speech_tokenizer(base_model)
        if speech_tokenizer is not None and getattr(speech_tokenizer, "model", None) is not None:
            speech_tokenizer.model.encoder.to(dtype=torch.float16)
            count += 1

        speaker_encoder = getattr(base_model.model, "speaker_encoder", None)
        if speaker_encoder is not None:
            speaker_encoder.to(dtype=torch.float16)
            count += 1

        talker.model.norm.to(dtype=torch.float16)
        count += 1
        talker.code_predictor.model.norm.to(dtype=torch.float16)
        count += 1

        return count

    @staticmethod
    def _apply_extra_precision(base_model, extra_precision: str, device: str, dtype: torch.dtype) -> int:
        """Apply validated fp16-inner wrappers outside decoder-layer islands."""
        mode = extra_precision.lower()
        if mode == "auto":
            # These include codec/logit heads and a partial predictor layer.
            # They can perturb token selection, so keep them stable by default.
            mode = "none"
        if mode in ("none", "off", "disabled"):
            return 0
        if mode != "fp16_inner":
            raise ValueError(
                f"Unsupported extra_precision {extra_precision!r}. Expected auto, none, or fp16_inner."
            )
        if dtype is not torch.float32:
            raise ValueError("extra_precision='fp16_inner' requires generation dtype fp32")

        from .mixed_precision import InnerDtypeLinear, replace_linear_modules

        talker = base_model.model.talker
        count = 0

        talker.codec_head = InnerDtypeLinear(talker.codec_head, torch.float16)
        count += 1

        count += replace_linear_modules(
            talker.text_projection,
            lambda _name, _linear: True,
            inner_dtype=torch.float16,
        )

        for idx, head in enumerate(talker.code_predictor.lm_head):
            talker.code_predictor.lm_head[idx] = InnerDtypeLinear(head, torch.float16)
            count += 1

        # Predictor layer 2 is content-sensitive as a full fp16 island and its
        # MLP also fails in fp16. The attention projections alone are stable as
        # fp16-inner linears and cover the largest safe remainder in that layer.
        predictor_layers = talker.code_predictor.model.layers
        if len(predictor_layers) > 2 and hasattr(predictor_layers[2], "self_attn"):
            layer = predictor_layers[2]
            layer.input_layernorm.to(dtype=torch.float16)
            count += 1
            layer.post_attention_layernorm.to(dtype=torch.float16)
            count += 1
            attn = layer.self_attn
            attn.q_norm.to(dtype=torch.float16)
            count += 1
            attn.k_norm.to(dtype=torch.float16)
            count += 1
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, name, InnerDtypeLinear(getattr(attn, name), torch.float16))
                count += 1

        return count

    @staticmethod
    def _patch_audio_frontend_dtype(base_model, audio_dtype: torch.dtype) -> None:
        """Move audio-side modules to audio_dtype and keep their inputs there.

        qwen-tts ties speech tokenizer and speaker encoder precision to the top-level
        model dtype. That breaks mixed precision: after moving speaker_encoder to
        fp32, upstream extract_speaker_embedding would still cast mels to the
        talker dtype. Patch that method locally so fp16 talker + fp32 audio modules
        works without modifying qwen-tts itself.
        """
        model = getattr(base_model, "model", None)
        if model is None:
            return

        speech_tokenizer = FasterQwen3TTS._get_speech_tokenizer(base_model)
        if speech_tokenizer is not None and getattr(speech_tokenizer, "model", None) is not None:
            speech_tokenizer.model.to(dtype=audio_dtype)

        speaker_encoder = getattr(model, "speaker_encoder", None)
        if speaker_encoder is None:
            return

        speaker_encoder.to(dtype=audio_dtype)

        @torch.inference_mode()
        def extract_speaker_embedding(self, audio, sr):
            assert sr == 24000, "Only support 24kHz audio"
            from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

            enc_param = next(self.speaker_encoder.parameters())
            mels = mel_spectrogram(
                torch.from_numpy(audio).unsqueeze(0),
                n_fft=1024,
                num_mels=128,
                sampling_rate=24000,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000,
            ).transpose(1, 2)
            mels = mels.to(device=enc_param.device, dtype=enc_param.dtype)
            return self.speaker_encoder(mels)[0]

        model.extract_speaker_embedding = MethodType(extract_speaker_embedding, model)

    @staticmethod
    def _resolve_non_streaming_mode(
        non_streaming_mode: Optional[bool],
        *,
        default: bool,
    ) -> bool:
        """Treat None as the method-specific upstream default."""
        return default if non_streaming_mode is None else non_streaming_mode
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: Union[str, torch.dtype, None] = "auto",
        audio_dtype: Union[str, torch.dtype, None] = "auto",
        layer_precision: str = "auto",
        predictor_layer_precision: str = "auto",
        audio_decoder_precision: str = "auto",
        large_block_precision: str = "auto",
        extra_precision: str = "auto",
        linear_precision: str = "none",
        attn_implementation: str = "auto",
        max_seq_len: int = 2048,
    ):
        """
        Load Qwen3-TTS model and prepare CUDA graphs.

        Args:
            model_name: Model path or HuggingFace Hub ID
            device: Device to use ("cuda" or "cpu")
            dtype: Data type for the autoregressive talker/predictor. "auto" uses
                fp32 on pre-Ampere CUDA GPUs such as V100, bf16 on native-bf16 GPUs,
                and fp32 on CPU. fp16 is available for experiments but can change
                generated codec tokens.
            audio_dtype: Data type for speech tokenizer and speaker encoder. "auto"
                keeps these modules fp32 when dtype resolves to fp16, because full
                fp16 decode can produce noisy audio.
            layer_precision: Select talker decoder layers to run as fp16 islands.
                "auto" wraps all talker decoder layers on V100 when dtype resolves
                to fp32. Also accepts "none", "all", "lastN", "firstN", "even",
                "odd", ranges like "14-27", or comma-separated indices.
            predictor_layer_precision: Select code-predictor decoder layers to run
                as fp16 islands. "auto" keeps predictor layers in the stable dtype;
                use an explicit selector such as "0,1,3,4" to opt in.
            audio_decoder_precision: Select speech-tokenizer decoder precision.
                "auto" keeps the tokenizer decoder in the stable dtype; pass "fp16"
                to opt into codec decoder fp16 islands.
            large_block_precision: Select large non-decoder blocks to keep in fp16.
                "auto" keeps embeddings, norms, speech-tokenizer encoder, and
                speaker encoder in the stable dtype; pass "fp16" to opt in.
            extra_precision: Apply validated fp16-inner wrappers to codec_head,
                text_projection, and code-predictor lm heads. "auto" keeps these
                logits/projection paths in the stable dtype.
            linear_precision: Optional inner precision for selected talker Linear
                layers. This is disabled by default because whole-layer fp16 islands
                are faster on V100.
            attn_implementation: Attention implementation. "auto" uses eager on
                pre-Ampere CUDA GPUs such as V100 and sdpa elsewhere.
            max_seq_len: Maximum sequence length for static cache
            
        Returns:
            FasterQwen3TTS instance
        """
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("CUDA graphs require CUDA device")

        dtype, audio_dtype = cls._resolve_dtypes(dtype, audio_dtype, device)
        attn_implementation = cls._resolve_attn_implementation(attn_implementation, device)
        
        logger.info(
            "Loading Qwen3-TTS model: %s (generation_dtype=%s, audio_dtype=%s, attn=%s)",
            model_name,
            dtype,
            audio_dtype,
            attn_implementation,
        )
        
        # Import here to avoid dependency issues (and suppress flash-attn warning)
        with suppress_flash_attn_warning():
            from qwen_tts import Qwen3TTSModel
        from .predictor_graph import PredictorGraph
        from .talker_graph import TalkerGraph
        # Load base model using qwen-tts library
        base_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        cls._patch_audio_frontend_dtype(base_model, audio_dtype)
        fp16_layer_indices = cls._apply_layer_precision(
            base_model,
            layer_precision,
            device,
            dtype,
        )
        fp16_predictor_layer_indices = cls._apply_predictor_layer_precision(
            base_model,
            predictor_layer_precision,
            device,
            dtype,
        )
        audio_decoder_precision_result = cls._apply_audio_decoder_precision(
            base_model,
            audio_decoder_precision,
            device,
            dtype,
        )
        large_block_wrappers = cls._apply_large_block_precision(
            base_model,
            large_block_precision,
            device,
            dtype,
        )
        extra_wrappers = cls._apply_extra_precision(
            base_model,
            extra_precision,
            device,
            dtype,
        )
        replaced_linears = cls._apply_linear_precision(
            base_model,
            linear_precision,
            device,
            dtype,
        )
        if replaced_linears:
            logger.info(
                "Applied %s inner-fp16 talker Linear wrappers",
                replaced_linears,
            )
        
        talker = base_model.model.talker
        talker_config = base_model.model.config.talker_config

        # Extract predictor config from loaded model
        predictor = talker.code_predictor
        pred_config = predictor.model.config
        talker_hidden = talker_config.hidden_size

        # Build CUDA graphs
        logger.info("Building CUDA graphs...")
        predictor_graph = PredictorGraph(
            predictor,
            pred_config,
            talker_hidden,
            device=device,
            dtype=dtype,
            do_sample=True,
            top_k=50,
            temperature=0.9,
        )
        for layer_idx in fp16_predictor_layer_indices:
            predictor_graph.set_layer_cache_dtype(layer_idx, torch.float16)
        if fp16_predictor_layer_indices:
            logger.info(
                "Applied fp16 predictor layer islands: %s",
                fp16_predictor_layer_indices,
            )
        if extra_wrappers:
            logger.info(
                "Applied %s extra fp16-inner wrappers",
                extra_wrappers,
            )
        if (
            audio_decoder_precision_result["transformer_layers"]
            or audio_decoder_precision_result["synthesis_wrappers"]
            or audio_decoder_precision_result["quantizer"]
            or audio_decoder_precision_result["projection_wrappers"]
        ):
            logger.info(
                "Applied fp16 audio decoder conversions: transformer_layers=%s, synthesis_wrappers=%s, quantizer=%s, projection_wrappers=%s",
                audio_decoder_precision_result["transformer_layers"],
                audio_decoder_precision_result["synthesis_wrappers"],
                audio_decoder_precision_result["quantizer"],
                audio_decoder_precision_result["projection_wrappers"],
            )
        if large_block_wrappers:
            logger.info(
                "Applied fp16 large-block conversions: %s",
                large_block_wrappers,
            )
        
        talker_graph = TalkerGraph(
            talker.model,
            talker_config,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
        for layer_idx in fp16_layer_indices:
            talker_graph.set_layer_cache_dtype(layer_idx, torch.float16)
        if fp16_layer_indices:
            logger.info(
                "Applied fp16 talker layer islands: %s",
                fp16_layer_indices,
            )
        
        logger.info("CUDA graphs initialized (will capture on first run)")
        
        return cls(
            base_model=base_model,
            predictor_graph=predictor_graph,
            talker_graph=talker_graph,
            device=device,
            dtype=dtype,
            audio_dtype=audio_dtype,
            max_seq_len=max_seq_len,
        )
    
    def _warmup(self, prefill_len: int):
        """Warm up and capture CUDA graphs with given prefill length."""
        if self._warmed_up:
            return
            
        logger.info("Warming up CUDA graphs...")
        self.predictor_graph.capture(num_warmup=3)
        self.talker_graph.capture(prefill_len=prefill_len, num_warmup=3)
        self._warmed_up = True
        logger.info("CUDA graphs captured and ready")
    
    def generate(
        self,
        text: str,
        language: str = "English",
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        """
        Generate speech from text using default voice.
        
        Not yet implemented - use generate_voice_clone() instead.
        """
        raise NotImplementedError(
            "Default voice generation not yet implemented. "
            "Use generate_voice_clone() with reference audio."
        )
    
    def _load_ref_audio_with_silence(self, ref_audio: Union[str, Path], silence_secs: float = 0.5) -> Tuple[np.ndarray, int]:
        """Load reference audio and optionally append trailing silence.

        The ICL voice-cloning prompt ends with the last codec token of the reference
        audio, so the model's first generated token is conditioned on whatever phoneme
        the reference ends with. Appending a short silence makes the last tokens
        encode silence instead, preventing that phoneme from bleeding into the start
        of the generated speech. Set silence_secs=0 to disable this behavior.
        """
        audio, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # convert to mono
        if silence_secs > 0:
            silence = np.zeros(int(silence_secs * sr), dtype=np.float32)
            audio = np.concatenate([audio, silence])
        return audio, sr

    def _resolve_voice_clone_prompt(
        self,
        input_ids,
        ref_audio: Optional[Union[str, Path]],
        ref_text: str,
        xvec_only: bool,
        append_silence: bool,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]],
    ) -> Tuple[Dict[str, Any], list, bool]:
        """Resolve voice clone prompt data and return (prompt, ref_ids, using_icl_mode)."""
        if voice_clone_prompt is not None:
            return self._resolve_precomputed_voice_clone_prompt(
                input_ids=input_ids,
                ref_text=ref_text,
                voice_clone_prompt=voice_clone_prompt,
            )
        if ref_audio is None:
            raise ValueError("ref_audio is required when voice_clone_prompt is not provided")

        return self._resolve_voice_clone_prompt_from_reference(
            input_ids=input_ids,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
        )

    def _resolve_precomputed_voice_clone_prompt(
        self,
        input_ids,
        ref_text: str,
        voice_clone_prompt: Union[Dict[str, Any], List[Any]],
    ) -> Tuple[Dict[str, Any], list, bool]:
        if isinstance(voice_clone_prompt, list):
            if len(voice_clone_prompt) != len(input_ids):
                raise ValueError(
                    f"voice_clone_prompt must have length {len(input_ids)}, got {len(voice_clone_prompt)}"
                )

            vcp = self.model._prompt_items_to_voice_clone_prompt(voice_clone_prompt)
            ref_ids = []
            for item in voice_clone_prompt:
                if bool(item.icl_mode):
                    item_ref_text = item.ref_text if item.ref_text else ref_text
                    if not item_ref_text:
                        raise ValueError(
                            "ref_text is required when voice_clone_prompt uses ICL mode."
                        )
                    ref_id = self.model._tokenize_texts(
                        [self.model._build_ref_text(item_ref_text)]
                    )[0]
                    ref_ids.append(ref_id)
                else:
                    ref_ids.append(None)

            return vcp, ref_ids, any(vcp["icl_mode"])

        required_keys = ("ref_spk_embedding",)
        missing = [k for k in required_keys if k not in voice_clone_prompt]
        if missing:
            raise ValueError(
                f"voice_clone_prompt missing required keys: {missing}. "
                f"Expected keys: {list(required_keys)}"
            )

        list_keys = ("ref_spk_embedding", "x_vector_only_mode", "icl_mode", "ref_code")
        for key in list_keys:
            if key not in voice_clone_prompt:
                continue
            value = voice_clone_prompt[key]
            if not isinstance(value, list) or len(value) != len(input_ids):
                raise ValueError(
                    f"voice_clone_prompt[{key!r}] must be a list with length {len(input_ids)}"
                )

        xvec_modes = voice_clone_prompt.get("x_vector_only_mode", [True] * len(input_ids))
        if "icl_mode" in voice_clone_prompt:
            icl_modes = [bool(v) for v in voice_clone_prompt["icl_mode"]]
            for i, (xvec_mode, icl_mode) in enumerate(zip(xvec_modes, icl_modes)):
                if bool(xvec_mode) == bool(icl_mode):
                    raise ValueError(
                        f"voice_clone_prompt has inconsistent mode flags at index {i}: "
                        "x_vector_only_mode and icl_mode must be opposites"
                    )
        else:
            icl_modes = [not bool(v) for v in xvec_modes]

        ref_codes = voice_clone_prompt.get("ref_code", [None] * len(input_ids))
        for i, (xvec_mode, icl_mode, ref_code) in enumerate(zip(xvec_modes, icl_modes, ref_codes)):
            if bool(xvec_mode) and ref_code is not None:
                raise ValueError(
                    f"voice_clone_prompt index {i}: ref_code must be None in x_vector_only mode"
                )
            if bool(icl_mode) and ref_code is None:
                raise ValueError(
                    f"voice_clone_prompt index {i}: ref_code is required in ICL mode"
                )

        vcp = dict(
            ref_code=ref_codes,
            ref_spk_embedding=voice_clone_prompt["ref_spk_embedding"],
            x_vector_only_mode=[bool(v) for v in xvec_modes],
            icl_mode=[bool(v) for v in icl_modes],
        )
        using_icl_mode = any(vcp["icl_mode"])

        if using_icl_mode:
            if not ref_text:
                raise ValueError(
                    "ref_text is required when voice_clone_prompt uses ICL mode."
                )
            ref_texts = [self.model._build_ref_text(ref_text)]
            # NOTE: single ref_text is shared across all ICL items in the batch.
            ref_id = self.model._tokenize_texts(ref_texts)[0]
            ref_ids = [ref_id if is_icl else None for is_icl in vcp["icl_mode"]]
        else:
            ref_ids = [None] * len(input_ids)

        return vcp, ref_ids, using_icl_mode

    def _resolve_voice_clone_prompt_from_reference(
        self,
        input_ids,
        ref_audio: Union[str, Path],
        ref_text: str,
        xvec_only: bool,
        append_silence: bool,
    ) -> Tuple[Dict[str, Any], list, bool]:
        using_icl_mode = not xvec_only
        cache_key = (str(ref_audio), ref_text, xvec_only, append_silence)
        cached_prompt = self._get_voice_prompt_cache(cache_key)
        if cached_prompt is not None:
            vcp, ref_ids = cached_prompt
            return vcp, ref_ids, using_icl_mode

        if xvec_only:
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text="",
                x_vector_only_mode=True,
            )
            spk_emb = prompt_items[0].ref_spk_embedding
            vcp = dict(
                ref_code=[None],
                ref_spk_embedding=[spk_emb],
                x_vector_only_mode=[True],
                icl_mode=[False],
            )
            ref_ids = [None] * len(input_ids)
            self._set_voice_prompt_cache(cache_key, (vcp, ref_ids))
            return vcp, ref_ids, using_icl_mode

        silence_secs = 0.5 if append_silence else 0.0
        ref_audio_input = self._load_ref_audio_with_silence(ref_audio, silence_secs=silence_secs)
        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio_input,
            ref_text=ref_text
        )
        vcp = self.model._prompt_items_to_voice_clone_prompt(prompt_items)

        ref_ids = []
        rt = prompt_items[0].ref_text
        if rt:
            ref_texts = [self.model._build_ref_text(rt)]
            ref_ids.append(self.model._tokenize_texts(ref_texts)[0])
        else:
            ref_ids.append(None)

        self._set_voice_prompt_cache(cache_key, (vcp, ref_ids))
        return vcp, ref_ids, using_icl_mode

    def _prepare_generation(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        language: str = "English",
        xvec_only: bool = False,
        non_streaming_mode: bool = False,
        append_silence: bool = True,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]] = None,
        instruct: Optional[str] = None,
    ):
        """Prepare inputs for generation (shared by streaming and non-streaming).

        Args:
            xvec_only: When True, use only the speaker embedding (x-vector) for voice
                cloning instead of the full ICL acoustic prompt. This prevents the model from
                continuing the reference audio's last phoneme and allows natural language switching.
                Default False to match upstream ICL behavior, where the full reference
                audio codec tokens are included in context.
            voice_clone_prompt: Optional precomputed prompt dict from
                `create_voice_clone_prompt`/`_prompt_items_to_voice_clone_prompt`.
                When provided, `xvec_only` is ignored. This path supports both:
                x-vector-only prompts (`ref_spk_embedding` only) and ICL prompts
                (`ref_spk_embedding` + `ref_code` + mode flags). `ref_text` is ignored
                for x-vector-only and required for ICL.
            instruct: Optional instruction string to guide generation style/language (e.g.
                "请用纯正广东话朗读"). Prepended as a user turn before the assistant TTS turn.
        """
        input_texts = [self.model._build_assistant_text(text)]
        input_ids = self.model._tokenize_texts(input_texts)

        instruct_ids = [None]
        if instruct:
            instruct_ids = [self.model._tokenize_texts([self.model._build_instruct_text(instruct)])[0]]

        vcp, ref_ids, using_icl_mode = self._resolve_voice_clone_prompt(
            input_ids=input_ids,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            append_silence=append_silence,
            voice_clone_prompt=voice_clone_prompt,
        )

        if instruct and not using_icl_mode:
            logger.warning(
                "Base-model instruct with x-vector-only voice cloning is experimental. "
                "Upstream Qwen3-TTS itself does not follow instructions reliably in this "
                "mode. Prefer xvec_only=False (ICL mode) when using instruct for voice "
                "cloning."
            )

        m = self.model.model

        tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=[language] if language is not None else ["Auto"],
            speakers=None,
            non_streaming_mode=non_streaming_mode,
            instruct_ids=instruct_ids,
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None

        # For ICL mode: return ref_codes so the decoder can use them as acoustic context
        ref_codes = None
        if using_icl_mode and vcp.get("ref_code") and vcp["ref_code"][0] is not None:
            ref_codes = vcp["ref_code"][0]

        return m, talker, config, tie, tam, tth, tpe, ref_codes

    def _prepare_generation_custom(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
    ):
        input_texts = [self.model._build_assistant_text(text)]
        input_ids = self.model._tokenize_texts(input_texts)

        instruct_ids = []
        if instruct is None or instruct == "":
            instruct_ids.append(None)
        else:
            instruct_ids.append(self.model._tokenize_texts([self.model._build_instruct_text(instruct)])[0])

        m = self.model.model
        tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=[None],
            voice_clone_prompt=None,
            languages=[language] if language is not None else ["Auto"],
            speakers=[speaker],
            non_streaming_mode=non_streaming_mode,
            instruct_ids=instruct_ids,
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None

        return m, talker, config, tie, tam, tth, tpe

    def _build_talker_inputs_local(
        self,
        m,
        input_ids,
        ref_ids,
        voice_clone_prompt,
        languages,
        speakers,
        non_streaming_mode: bool,
        instruct_ids=None,
    ):
        """Local copy of upstream talker input building for qwen-tts main repo."""
        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = m.generate_speaker_prompt(voice_clone_prompt)

        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        m.talker.text_projection(m.talker.get_text_embeddings()(instruct_id))
                    )

        if speakers is None:
            speakers = [None] * len(input_ids)

        trailing_text_hiddens = []
        tts_pad_embed = None

        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:
                    speaker_embed = None
                else:
                    if speaker.lower() not in m.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    spk_id = m.config.talker_config.spk_id[speaker.lower()]
                    speaker_embed = m.talker.get_input_embeddings()(
                        torch.tensor(spk_id, device=m.talker.device, dtype=input_id.dtype)
                    )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None
            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in m.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                language_id = m.config.talker_config.codec_language_id[language.lower()]

            if (
                language.lower() in ["chinese", "auto"]
                and speaker not in ("", None)
                and m.config.talker_config.spk_is_dialect[speaker.lower()]
            ):
                dialect = m.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = m.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = m.talker.text_projection(
                m.talker.get_text_embeddings()(
                    torch.tensor(
                        [[m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]],
                        device=m.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)

            if language_id is None:
                codec_prefill_list = [[
                    m.config.talker_config.codec_nothink_id,
                    m.config.talker_config.codec_think_bos_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]
            else:
                codec_prefill_list = [[
                    m.config.talker_config.codec_think_id,
                    m.config.talker_config.codec_think_bos_id,
                    language_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]

            codec_input_emebdding_0 = m.talker.get_input_embeddings()(
                torch.tensor(codec_prefill_list, device=m.talker.device, dtype=input_id.dtype)
            )
            codec_input_emebdding_1 = m.talker.get_input_embeddings()(
                torch.tensor(
                    [[m.config.talker_config.codec_pad_id, m.config.talker_config.codec_bos_id]],
                    device=m.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1], dim=1)

            _talker_input_embed_role = m.talker.text_projection(
                m.talker.get_text_embeddings()(input_id[:, :3])
            )
            _talker_input_embed = torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            ) + codec_input_emebdding[:, :-1]

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            if (
                voice_clone_prompt is not None
                and voice_clone_prompt.get("ref_code", None) is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_input_embed, trailing_text_hidden = m.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(m.talker.device).clone(),  # escape inference_mode context
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        m.talker.text_projection(
                            m.talker.get_text_embeddings()(input_id[:, 3:4])
                        )
                        + codec_input_emebdding[:, -1:],
                    ],
                    dim=1,
                )
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            torch.cat(
                                (
                                    m.talker.text_projection(
                                        m.talker.get_text_embeddings()(input_id[:, 3:-5])
                                    ),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + m.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[m.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                    device=m.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + m.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[m.config.talker_config.codec_bos_id]],
                                    device=m.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    trailing_text_hidden = torch.cat(
                        (
                            m.talker.text_projection(
                                m.talker.get_text_embeddings()(input_id[:, 4:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )

            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed,
            batch_first=True,
            padding_value=0.0,
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])

        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)

        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad,
            batch_first=True,
            padding_value=0.0,
        )
        arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
            len(trailing_text_original_lengths), -1
        )
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        return talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed

    @torch.inference_mode()
    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        xvec_only: bool = False,
        non_streaming_mode: Optional[bool] = None,
        append_silence: bool = True,
        instruct: Optional[str] = None,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> Tuple[list, int]:
        """
        Generate speech with voice cloning using reference audio.

        Args:
            text: Text to synthesize
            language: Target language
            ref_audio: Path to reference audio file. Required when `voice_clone_prompt` is not provided.
            ref_text: Transcription of reference audio.
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens before EOS is allowed
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty
            xvec_only: When True, use only the speaker embedding for voice cloning.
                This prevents phoneme bleed-through from the reference and allows clean
                language switching. Default False to match upstream ICL behavior
                (reference audio in context).
            non_streaming_mode: Match upstream text-feeding layout. When None, use the
                upstream voice-cloning default (False, step-by-step text feeding during
                decode). Set True to prefill the full target text before decode.
            voice_clone_prompt: Optional precomputed voice clone prompt dict. When provided,
                `xvec_only` is ignored and prompt extraction from `ref_audio` is skipped.
                This path supports x-vector-only prompts (`ref_spk_embedding` only)
                and ICL prompts (`ref_spk_embedding` + `ref_code` + mode flags).
                `ref_text` is ignored for x-vector-only and required for ICL.
            instruct: Optional instruction to guide generation style/dialect (e.g.
                "请用纯正广东话朗读"). Prepended as a user turn before the TTS assistant turn.
                Experimental for x-vector-only voice cloning; prefer `xvec_only=False`.

        Returns:
            Tuple of ([audio_waveform], sample_rate)
        """
        from .generate import fast_generate

        non_streaming_mode = self._resolve_non_streaming_mode(
            non_streaming_mode,
            default=False,
        )

        m, talker, config, tie, tam, tth, tpe, ref_codes = self._prepare_generation(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            non_streaming_mode=non_streaming_mode,
            append_silence=append_silence,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
        )

        codec_ids, timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

        if codec_ids is None:
            logger.warning("Generation returned no tokens")
            return [np.zeros(1, dtype=np.float32)], self.sample_rate

        # In ICL mode: prepend reference codes before decoding so the codec decoder
        # has acoustic context from the reference audio (matches official implementation).
        speech_tokenizer = m.speech_tokenizer
        if ref_codes is not None:
            ref_codes_dev = ref_codes.to(codec_ids.device)
            codes_for_decode = torch.cat([ref_codes_dev, codec_ids], dim=0)
        else:
            codes_for_decode = codec_ids
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_for_decode.unsqueeze(0)})

        # Convert to numpy and trim off the reference audio portion
        ref_len = ref_codes.shape[0] if ref_codes is not None else 0
        total_len = codes_for_decode.shape[0]
        audio_arrays = []
        for a in audio_list:
            if hasattr(a, 'cpu'):  # torch tensor
                a = a.flatten().cpu().numpy()
            else:  # already numpy
                a = a.flatten() if hasattr(a, 'flatten') else a
            if ref_len > 0:
                cut = int(ref_len / max(total_len, 1) * len(a))
                a = a[cut:]
            audio_arrays.append(a)
        
        n_steps = timing['steps']
        audio_duration = n_steps / 12.0  # 12 Hz codec
        total_time = timing['prefill_ms']/1000 + timing['decode_s']
        rtf = audio_duration / total_time if total_time > 0 else 0
        
        logger.info(
            f"Generated {audio_duration:.2f}s audio in {total_time:.2f}s "
            f"({timing['ms_per_step']:.1f}ms/step, RTF: {rtf:.2f})"
        )
        
        return audio_arrays, sr

    @torch.inference_mode()
    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        xvec_only: bool = False,
        non_streaming_mode: Optional[bool] = None,
        append_silence: bool = True,
        parity_mode: bool = False,
        instruct: Optional[str] = None,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        """
        Stream voice-cloned speech generation, yielding audio chunks.

        Same as generate_voice_clone() but yields (audio_chunk, sample_rate, timing)
        tuples every chunk_size codec steps (~chunk_size/12 seconds of audio).

        Args:
            text: Text to synthesize
            language: Target language
            ref_audio: Path to reference audio file. Required when `voice_clone_prompt` is not provided.
            ref_text: Transcription of reference audio.
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens before EOS is allowed
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty
            chunk_size: Codec steps per chunk (12 = ~1 second)
            xvec_only: When True, use only the speaker embedding for voice cloning.
                This prevents phoneme bleed-through from the reference and allows clean
                language switching. Default False to match upstream ICL behavior
                (reference audio in context).
            non_streaming_mode: When None, use the upstream voice-cloning default
                (False, step-by-step text feeding during decode). Set to True to
                prefill the full target text before streaming decode.
            parity_mode: When True, disables CUDA graphs and uses dynamic cache streaming.
            voice_clone_prompt: Optional precomputed voice clone prompt dict. When provided,
                `xvec_only` is ignored and prompt extraction from `ref_audio` is skipped.
                This path supports x-vector-only prompts (`ref_spk_embedding` only)
                and ICL prompts (`ref_spk_embedding` + `ref_code` + mode flags).
                `ref_text` is ignored for x-vector-only and required for ICL.
            instruct: Optional instruction to guide generation style/dialect (e.g.
                "请用纯正广东话朗读"). Prepended as a user turn before the TTS assistant turn.
                Experimental for x-vector-only voice cloning; prefer `xvec_only=False`.

        Yields:
            Tuple of (audio_chunk_numpy, sample_rate, timing_dict)
        """
        from .streaming import fast_generate_streaming, parity_generate_streaming

        non_streaming_mode = self._resolve_non_streaming_mode(
            non_streaming_mode,
            default=False,
        )

        m, talker, config, tie, tam, tth, tpe, ref_codes = self._prepare_generation(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            non_streaming_mode=non_streaming_mode,
            append_silence=append_silence,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
        )

        speech_tokenizer = m.speech_tokenizer

        # Hybrid decode strategy:
        # 1. Accumulated decode for early chunks (correct, calibrates samples_per_frame)
        # 2. Sliding window with 25-frame left context once calibrated (constant cost)
        # This avoids boundary artifacts (pops) while keeping decode cost bounded.
        context_frames = 25
        min_calibration_frames = max(context_frames, chunk_size)
        all_codes = []
        prev_gen_audio_len = 0  # tracks position within the generated (non-ref) audio
        samples_per_frame = None

        stream_fn = parity_generate_streaming if parity_mode else fast_generate_streaming
        stream_kwargs = dict(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        )
        if not parity_mode:
            stream_kwargs["predictor_graph"] = self.predictor_graph
            stream_kwargs["talker_graph"] = self.talker_graph

        for codec_chunk, timing in stream_fn(**stream_kwargs):
            all_codes.append(codec_chunk)
            n_new = codec_chunk.shape[0]
            all_flat = torch.cat(all_codes, dim=0)
            n_total = all_flat.shape[0]

            if samples_per_frame is None:
                # Phase 1: accumulated decode until we can calibrate.
                # In ICL mode prepend reference codes so the codec decoder has acoustic
                # context from the reference audio (matches official implementation).
                if ref_codes is not None:
                    codes_input = torch.cat([ref_codes.to(all_flat.device), all_flat], dim=0)
                else:
                    codes_input = all_flat
                audio_list, sr = speech_tokenizer.decode(
                    {"audio_codes": codes_input.unsqueeze(0)}
                )
                audio = audio_list[0]
                if hasattr(audio, 'cpu'):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, 'flatten') else audio

                # Separate out reference audio portion; track position in generated audio only
                if ref_codes is not None:
                    ref_len = ref_codes.shape[0]
                    total_len = codes_input.shape[0]
                    ref_audio_cut = int(ref_len / max(total_len, 1) * len(audio))
                    gen_audio = audio[ref_audio_cut:]
                else:
                    gen_audio = audio

                new_audio = gen_audio[prev_gen_audio_len:]
                prev_gen_audio_len = len(gen_audio)

                if n_total >= min_calibration_frames:
                    samples_per_frame = len(gen_audio) / n_total
            else:
                # Phase 2: sliding window with left context
                ctx_start = max(0, n_total - n_new - context_frames)
                window = all_flat[ctx_start:]
                n_ctx = window.shape[0] - n_new

                audio_list, sr = speech_tokenizer.decode(
                    {"audio_codes": window.unsqueeze(0)}
                )
                audio = audio_list[0]
                if hasattr(audio, 'cpu'):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, 'flatten') else audio

                if n_ctx > 0:
                    ctx_samples = int(round(n_ctx * samples_per_frame))
                    new_audio = audio[ctx_samples:]
                else:
                    new_audio = audio

            yield new_audio, sr, timing

    @torch.inference_mode()
    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        if self.model.model.tts_model_type != "custom_voice":
            raise ValueError("Loaded model does not support custom voice generation")

        self.model._validate_languages([language])
        self.model._validate_speakers([speaker])

        non_streaming_mode = self._resolve_non_streaming_mode(
            non_streaming_mode,
            default=True,
        )

        if self.model.model.tts_model_size in "0b6":
            instruct = None

        from .generate import fast_generate

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

        codec_ids, timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

        if codec_ids is None:
            logger.warning("Generation returned no tokens")
            return [np.zeros(1, dtype=np.float32)], self.sample_rate

        speech_tokenizer = m.speech_tokenizer
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codec_ids.unsqueeze(0)})

        audio_arrays = []
        for a in audio_list:
            if hasattr(a, "cpu"):
                audio_arrays.append(a.flatten().cpu().numpy())
            else:
                audio_arrays.append(a.flatten() if hasattr(a, "flatten") else a)

        n_steps = timing["steps"]
        audio_duration = n_steps / 12.0
        total_time = timing["prefill_ms"] / 1000 + timing["decode_s"]
        rtf = audio_duration / total_time if total_time > 0 else 0

        logger.info(
            f"Generated {audio_duration:.2f}s audio in {total_time:.2f}s "
            f"({timing['ms_per_step']:.1f}ms/step, RTF: {rtf:.2f})"
        )

        return audio_arrays, sr

    @torch.inference_mode()
    def generate_custom_voice_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        if self.model.model.tts_model_type != "custom_voice":
            raise ValueError("Loaded model does not support custom voice generation")

        self.model._validate_languages([language])
        self.model._validate_speakers([speaker])

        non_streaming_mode = self._resolve_non_streaming_mode(
            non_streaming_mode,
            default=True,
        )

        if self.model.model.tts_model_size in "0b6":
            instruct = None

        from .streaming import fast_generate_streaming

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

        speech_tokenizer = m.speech_tokenizer

        context_frames = 25
        min_calibration_frames = max(context_frames, chunk_size)
        all_codes = []
        prev_audio_len = 0
        samples_per_frame = None

        for codec_chunk, timing in fast_generate_streaming(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        ):
            all_codes.append(codec_chunk)
            n_new = codec_chunk.shape[0]
            all_flat = torch.cat(all_codes, dim=0)
            n_total = all_flat.shape[0]

            if samples_per_frame is None:
                audio_list, sr = speech_tokenizer.decode({"audio_codes": all_flat.unsqueeze(0)})
                audio = audio_list[0]
                if hasattr(audio, "cpu"):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, "flatten") else audio

                new_audio = audio[prev_audio_len:]
                prev_audio_len = len(audio)

                if n_total >= min_calibration_frames:
                    samples_per_frame = len(audio) / n_total
            else:
                ctx_start = max(0, n_total - n_new - context_frames)
                window = all_flat[ctx_start:]
                n_ctx = window.shape[0] - n_new

                audio_list, sr = speech_tokenizer.decode({"audio_codes": window.unsqueeze(0)})
                audio = audio_list[0]
                if hasattr(audio, "cpu"):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, "flatten") else audio

                if n_ctx > 0:
                    ctx_samples = int(round(n_ctx * samples_per_frame))
                    new_audio = audio[ctx_samples:]
                else:
                    new_audio = audio

            yield new_audio, sr, timing

    @torch.inference_mode()
    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        if self.model.model.tts_model_type != "voice_design":
            raise ValueError("Loaded model does not support voice design generation")

        self.model._validate_languages([language])

        non_streaming_mode = self._resolve_non_streaming_mode(
            non_streaming_mode,
            default=True,
        )

        from .generate import fast_generate

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text,
            language=language,
            speaker=None,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

        codec_ids, timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

        if codec_ids is None:
            logger.warning("Generation returned no tokens")
            return [np.zeros(1, dtype=np.float32)], self.sample_rate

        speech_tokenizer = m.speech_tokenizer
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codec_ids.unsqueeze(0)})

        audio_arrays = []
        for a in audio_list:
            if hasattr(a, "cpu"):
                audio_arrays.append(a.flatten().cpu().numpy())
            else:
                audio_arrays.append(a.flatten() if hasattr(a, "flatten") else a)

        n_steps = timing["steps"]
        audio_duration = n_steps / 12.0
        total_time = timing["prefill_ms"] / 1000 + timing["decode_s"]
        rtf = audio_duration / total_time if total_time > 0 else 0

        logger.info(
            f"Generated {audio_duration:.2f}s audio in {total_time:.2f}s "
            f"({timing['ms_per_step']:.1f}ms/step, RTF: {rtf:.2f})"
        )

        return audio_arrays, sr

    @torch.inference_mode()
    def generate_voice_design_streaming(
        self,
        text: str,
        instruct: str,
        language: str,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        if self.model.model.tts_model_type != "voice_design":
            raise ValueError("Loaded model does not support voice design generation")

        self.model._validate_languages([language])

        non_streaming_mode = self._resolve_non_streaming_mode(
            non_streaming_mode,
            default=True,
        )

        from .streaming import fast_generate_streaming

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text,
            language=language,
            speaker=None,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

        speech_tokenizer = m.speech_tokenizer

        context_frames = 25
        min_calibration_frames = max(context_frames, chunk_size)
        all_codes = []
        prev_audio_len = 0
        samples_per_frame = None

        for codec_chunk, timing in fast_generate_streaming(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        ):
            all_codes.append(codec_chunk)
            n_new = codec_chunk.shape[0]
            all_flat = torch.cat(all_codes, dim=0)
            n_total = all_flat.shape[0]

            if samples_per_frame is None:
                audio_list, sr = speech_tokenizer.decode({"audio_codes": all_flat.unsqueeze(0)})
                audio = audio_list[0]
                if hasattr(audio, "cpu"):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, "flatten") else audio

                new_audio = audio[prev_audio_len:]
                prev_audio_len = len(audio)

                if n_total >= min_calibration_frames:
                    samples_per_frame = len(audio) / n_total
            else:
                ctx_start = max(0, n_total - n_new - context_frames)
                window = all_flat[ctx_start:]
                n_ctx = window.shape[0] - n_new

                audio_list, sr = speech_tokenizer.decode({"audio_codes": window.unsqueeze(0)})
                audio = audio_list[0]
                if hasattr(audio, "cpu"):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, "flatten") else audio

                if n_ctx > 0:
                    ctx_samples = int(round(n_ctx * samples_per_frame))
                    new_audio = audio[ctx_samples:]
                else:
                    new_audio = audio

            yield new_audio, sr, timing
