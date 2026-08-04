"""Seed-VC request models and backend implementation."""

from __future__ import annotations

import base64
import logging
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
import numpy as np
from pydantic import BaseModel, Field

from .audio_utils import (
    MP3_INTERMEDIATE_WAV_SUBTYPE,
    _audio_file_to_mp3_bytes,
    _audio_to_float32,
    _audio_to_wav_bytes,
    _safe_file_stem,
    _temporary_cwd,
)

_LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class SeedVCRequest(BaseModel):
    """Request body compatible with the Seed-VC /vc endpoint."""

    audio: str = Field(..., description="Base64 encoded source audio")
    reference_url: Optional[str] = Field(None, description="Reference voice URL; optional when cached sample/embedding exists")
    id: str = Field(..., description="Voice id used for cached reference audio/embeddings")
    style: Optional[str] = "general"
    intensity: Optional[float] = 1.0
    preset: Optional[str] = None
    remove_glitches: Optional[bool] = False


class SeedVCBatchRequest(BaseModel):
    """Batch Seed-VC request.

    The initial batched path is intended for a Starling -> one target timbre
    pipeline, so all items share the same target voice settings.
    """

    items: list[SeedVCRequest] = Field(..., min_length=1)
    max_chunk_batch_size: int = Field(1, ge=1, le=64)


class SeedVCFindVoiceRequest(BaseModel):
    reference_url: str
    id: str


class SeedVCEnhanceRequest(BaseModel):
    reference_url: str
    id: str


class SeedVCBackend:
    """Embedded Seed-VC inference backend compatible with the standalone /vc API."""

    model_presets = {
        "default": {"diffusion_steps": 30},
        "distilled": {"diffusion_steps": 30},
        "medium": {"diffusion_steps": 23},
        "fast": {"diffusion_steps": 15},
    }

    def __init__(self, settings: Any):
        self.settings = settings
        self.runtime_root = _resolve_project_path(settings.runtime_root).resolve()
        self.root = _resolve_project_path(settings.root).resolve()
        self.tmp_dir = _resolve_project_path(settings.tmp_dir).resolve()
        self.output_dir = _resolve_project_path(settings.output_dir).resolve()
        self.voice_samples_dir = _resolve_project_path(settings.voice_samples_dir).resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.runtime_root.exists():
            raise FileNotFoundError(f"Seed-VC runtime root not found: {self.runtime_root}")
        if not self.root.exists():
            raise FileNotFoundError(f"Seed-VC asset root not found: {self.root}")
        if str(self.runtime_root) not in sys.path:
            sys.path.insert(0, str(self.runtime_root))

        import torch  # pylint: disable=import-outside-toplevel
        import torchaudio  # pylint: disable=import-outside-toplevel
        import yaml  # pylint: disable=import-outside-toplevel
        from transformers import AutoFeatureExtractor, WhisperModel  # pylint: disable=import-outside-toplevel

        with _temporary_cwd(self.root):
            from hf_utils import load_custom_model_from_hf  # pylint: disable=import-outside-toplevel
            from inference import convert_voice, crossfade, find_silence_boundaries  # pylint: disable=import-outside-toplevel
            from modules.audio import mel_spectrogram  # pylint: disable=import-outside-toplevel
            from modules.bigvgan import bigvgan  # pylint: disable=import-outside-toplevel
            from modules.campplus.DTDNN import CAMPPlus  # pylint: disable=import-outside-toplevel
            from modules.commons import build_model, load_checkpoint, recursive_munch  # pylint: disable=import-outside-toplevel
            from modules.lazy_embedding_loader import HDF5EmbeddingLoader  # pylint: disable=import-outside-toplevel

            self.torch = torch
            self.torchaudio = torchaudio
            self.convert_voice = convert_voice
            self.seed_vc_crossfade = crossfade
            self.find_silence_boundaries = find_silence_boundaries

            if torch.cuda.is_available() and self.settings.device.startswith("cuda"):
                self.device = torch.device(self.settings.device)
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.dtype = torch.float16 if self.settings.fp16 else torch.float32

            _LOGGER.info("Loading Seed-VC models on %s from %s", self.device, self.root)
            medium_checkpoint_path, medium_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            )
            with open(medium_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            self.config = config
            self.model_params = recursive_munch(config["model_params"])
            self.model_params.dit_type = "DiT"
            self.sample_rate = int(config["preprocess_params"]["sr"])
            self.hop_length = int(config["preprocess_params"]["spect_params"]["hop_length"])

            self.default_model = build_model(self.model_params, stage="DiT")
            self.default_model, _, _, _ = load_checkpoint(
                self.default_model,
                None,
                medium_checkpoint_path,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
            self._prepare_model_dict(self.default_model)
            self.model_cache = {"default": self.default_model}

            campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu", weights_only=False))
            self.campplus_model.eval().to(self.device)

            whisper_name = self.model_params.speech_tokenizer.name
            self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
            del self.whisper_model.decoder
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(self.model_params.vocoder.name, use_cuda_kernel=False)
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval().to(self.device)

            spect_params = config["preprocess_params"]["spect_params"]
            mel_fn_args = {
                "n_fft": spect_params["n_fft"],
                "win_size": spect_params["win_length"],
                "hop_size": spect_params["hop_length"],
                "num_mels": spect_params["n_mels"],
                "sampling_rate": self.sample_rate,
                "fmin": spect_params.get("fmin", 0),
                "fmax": None,
                "center": False,
            }
            self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

            embeddings_path = _resolve_project_path(self.settings.embeddings_hdf5_path)
            if not embeddings_path.is_file():
                raise RuntimeError(f"Seed-VC embeddings file not found: {embeddings_path}")
            self.cached_embeddings = HDF5EmbeddingLoader(
                embeddings_path,
                cache_size=self.settings.embedding_cache_size,
            )

            try:
                from find_voice import find_base_voice  # pylint: disable=import-outside-toplevel
            except Exception as exc:  # pylint: disable=broad-exception-caught
                find_base_voice = None
                _LOGGER.warning("Seed-VC find_voice unavailable: %s", exc)
            try:
                from glitch_remover import process_file  # pylint: disable=import-outside-toplevel
            except Exception as exc:  # pylint: disable=broad-exception-caught
                process_file = None
                _LOGGER.warning("Seed-VC glitch remover unavailable: %s", exc)

            self.find_base_voice = find_base_voice
            self.process_file = process_file
            _LOGGER.info(
                "Seed-VC backend ready: sr=%d cached_embeddings=%d",
                self.sample_rate,
                len(self.cached_embeddings),
            )

    def _prepare_model_dict(self, model_dict: dict[str, Any]) -> None:
        for key in model_dict:
            model_dict[key].eval()
            model_dict[key].to(self.device)
        estimator = getattr(model_dict.cfm, "estimator", None)
        if estimator is not None and hasattr(estimator, "setup_caches"):
            estimator.setup_caches(
                max_batch_size=self.settings.estimator_cache_batch_size,
                max_seq_length=self.settings.estimator_cache_seq_length,
            )

    def get_semantic_features(self, waves_16k):
        torch = self.torch
        ori_inputs = self.whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        ori_input_features = self.whisper_model._mask_input_features(
            ori_inputs.input_features,
            attention_mask=ori_inputs.attention_mask,
        ).to(self.device)
        with torch.no_grad():
            ori_outputs = self.whisper_model.encoder(ori_input_features.to(self.whisper_model.encoder.dtype))
        features = ori_outputs.last_hidden_state.to(torch.float32)
        return features[:, : waves_16k.size(-1) // 320 + 1]

    def get_semantic_features_batch(self, waves_16k: list[Any]):
        torch = self.torch
        lengths = [int(wave.size(-1)) for wave in waves_16k]
        ori_inputs = self.whisper_feature_extractor(
            [wave.squeeze(0).cpu().numpy() for wave in waves_16k],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        ori_input_features = self.whisper_model._mask_input_features(
            ori_inputs.input_features,
            attention_mask=ori_inputs.attention_mask,
        ).to(self.device)
        with torch.no_grad():
            ori_outputs = self.whisper_model.encoder(ori_input_features.to(self.whisper_model.encoder.dtype))
        features = ori_outputs.last_hidden_state.to(torch.float32)
        max_frames = max(length // 320 + 1 for length in lengths)
        return features[:, :max_frames], torch.LongTensor([length // 320 + 1 for length in lengths]).to(self.device)

    async def _fetch_sample(self, request: SeedVCRequest) -> Path:
        sample_path = self.voice_samples_dir / f"{_safe_file_stem(request.id)}.mp3"
        if sample_path.exists():
            return sample_path
        if not request.reference_url:
            raise ValueError(f"No cached sample for voice {request.id!r} and no reference_url provided")
        _LOGGER.info("Fetching Seed-VC reference sample: id=%s url=%s", request.id, request.reference_url)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(request.reference_url)
            resp.raise_for_status()
            sample_path.write_bytes(resp.content)
        return sample_path

    @staticmethod
    def _embedding_key(voice_id: str, style: str, intensity: float) -> str:
        if style == "general":
            return f"{voice_id}.general"
        if intensity == 1.0:
            return f"{voice_id}.{style}"
        if intensity == int(intensity):
            return f"{voice_id}.{style}.{int(intensity)}"
        return f"{voice_id}.{style}.{intensity}"

    def _available_styles_for_voice(self, voice_id: str) -> dict[str, list[float]]:
        styles: dict[str, set[float]] = {}
        if not self.cached_embeddings:
            return {}
        prefix = f"{voice_id}."
        for key in self.cached_embeddings.keys():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            style = suffix
            intensity = 1.0
            parts = suffix.split(".")
            for idx in range(1, len(parts)):
                try:
                    intensity = float(".".join(parts[idx:]))
                    style = ".".join(parts[:idx])
                    break
                except ValueError:
                    pass
            styles.setdefault(style, set()).add(intensity)
        return {style: sorted(intensities) for style, intensities in sorted(styles.items())}

    def _resolve_exact_cached_embeddings(self, voice_id: str, style: str, intensity: float) -> tuple[str, Any]:
        embedding_key = self._embedding_key(voice_id, style, intensity)
        if style == "general" and intensity != 1.0:
            embedding_key = f"{voice_id}.general.{intensity:g}"
        cached = self.cached_embeddings.get(embedding_key) if self.cached_embeddings else None
        if cached is not None:
            return embedding_key, cached

        available = self._available_styles_for_voice(voice_id)
        if not available:
            raise ValueError(f"No cached Seed-VC embeddings found for voice {voice_id!r}")
        if style not in available:
            raise ValueError(
                f"Unsupported style {style!r} for voice {voice_id!r}; supported styles: {sorted(available)}"
            )
        raise ValueError(
            f"Unsupported styleIntensity {intensity:g} for voice {voice_id!r} style {style!r}; "
            f"supported intensities: {available[style]}"
        )

    def _resolve_cached_embeddings(self, request: SeedVCRequest) -> tuple[str, Any | None]:
        style = request.style or "general"
        intensity = request.intensity or 1.0
        embedding_key = self._embedding_key(request.id, style, intensity)
        cached = self.cached_embeddings.get(embedding_key) if self.cached_embeddings else None
        if cached or style == "general" or not self.cached_embeddings:
            return embedding_key, cached

        available: list[tuple[float, str]] = []
        prefix = f"{request.id}.{style}"
        for key in self.cached_embeddings.keys():
            if key == prefix:
                available.append((1.0, key))
            elif key.startswith(prefix + "."):
                try:
                    available.append((float(key[len(prefix) + 1 :]), key))
                except ValueError:
                    pass
        if not available:
            return embedding_key, None
        available.sort(key=lambda item: abs(item[0] - intensity))
        closest_intensity, closest_key = available[0]
        _LOGGER.info("Seed-VC exact intensity %.3f not found for %s; using %.3f", intensity, prefix, closest_intensity)
        return closest_key, self.cached_embeddings.get(closest_key)

    def _convert_voice_v1(
        self,
        source_path: Path,
        target_path: Path | None,
        preset_config: dict[str, Any],
        cached_embeddings: Any | None,
        voice_id: str | None,
    ) -> Path:
        torch = self.torch
        import soundfile as sf  # pylint: disable=import-outside-toplevel

        diffusion_steps = int(preset_config.get("diffusion_steps", 25))
        length_adjust = float(preset_config.get("length_adjust", 1.0))
        cfg_rate = float(preset_config.get("cfg_rate", 0.7))
        model = self.model_cache["default"]

        with torch.no_grad(), _temporary_cwd(self.root):
            vc_wave = self.convert_voice(
                source_audio_path=str(source_path),
                ref_audio_path=str(target_path) if target_path is not None else None,
                model=model,
                semantic_fn=self.get_semantic_features,
                vocoder_fn=self.bigvgan_model,
                campplus_model=self.campplus_model,
                mel_fn=self.to_mel,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=cfg_rate,
                f0_condition=False,
                f0_fn=None,
                device=self.device,
                fp16=self.settings.fp16,
                cached_ref_embeddings=cached_embeddings,
            )

        source_name = source_path.stem
        target_name = _safe_file_stem(voice_id or (target_path.stem if target_path is not None else "cached"))
        output_path = self.output_dir / f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{cfg_rate}.wav"
        if torch.is_tensor(vc_wave):
            vc_wave = vc_wave.detach().float().cpu().numpy()
        vc_wave = np.asarray(vc_wave, dtype=np.float32).squeeze()
        sf.write(output_path, vc_wave, self.sample_rate, subtype=MP3_INTERMEDIATE_WAV_SUBTYPE)
        return output_path

    def _prepare_seed_vc_reference(self, target_path: Path | None, cached_ref_embeddings: Any | None, model: Any):
        torch = self.torch
        if cached_ref_embeddings is not None:
            style = cached_ref_embeddings["style"].to(self.device)
            mel_ref = cached_ref_embeddings["mel_ref"].to(self.device)
            prompt_condition = cached_ref_embeddings["prompt_condition"].to(self.device)
            if style.dim() == 1:
                style = style.unsqueeze(0)
            if mel_ref.dim() == 2:
                mel_ref = mel_ref.unsqueeze(0)
            if prompt_condition.dim() == 2:
                prompt_condition = prompt_condition.unsqueeze(0)
            return style, mel_ref, prompt_condition

        if target_path is None:
            raise ValueError("target_path is required when cached reference embeddings are unavailable")

        import librosa  # pylint: disable=import-outside-toplevel

        ref_audio_np = librosa.load(target_path, sr=self.sample_rate)[0]
        ref_audio = torch.tensor(ref_audio_np[: self.sample_rate * 25]).unsqueeze(0).float().to(self.device)
        ref_16k = self.torchaudio.functional.resample(ref_audio, self.sample_rate, 16000)
        semantic_ref = self.get_semantic_features(ref_16k)
        mel_ref = self.to_mel(ref_audio.float())
        ref_lengths = torch.LongTensor([mel_ref.size(2)]).to(self.device)
        feat_ref = self.torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat_ref = feat_ref - feat_ref.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat_ref.unsqueeze(0))
        prompt_condition, _, _, _, _ = model.length_regulator(
            semantic_ref, ylens=ref_lengths, n_quantizers=3, f0=None
        )
        return style, mel_ref, prompt_condition

    @staticmethod
    def _fade_seed_vc_start(result: np.ndarray, sample_rate: int) -> np.ndarray:
        check_window = int(sample_rate * 0.03)
        if len(result) <= check_window:
            return result
        first_10ms = int(sample_rate * 0.01)
        window_20_30ms = result[int(sample_rate * 0.02) : check_window]
        energy_first = np.sqrt(np.mean(result[:first_10ms] ** 2))
        energy_later = np.sqrt(np.mean(window_20_30ms ** 2))
        if energy_first > energy_later * 1.5:
            fade_samples = int(sample_rate * 0.025)
            fade_in = np.linspace(0, 1, fade_samples) ** 4
        else:
            fade_samples = int(sample_rate * 0.005)
            fade_in = np.linspace(0, 1, fade_samples)
        result[:fade_samples] *= fade_in
        return result

    def _convert_voice_v1_batch(
        self,
        source_paths: list[Path],
        target_path: Path | None,
        preset_config: dict[str, Any],
        cached_embeddings: Any | None,
        max_chunk_batch_size: int,
    ) -> list[np.ndarray]:
        torch = self.torch
        import librosa  # pylint: disable=import-outside-toplevel

        diffusion_steps = int(preset_config.get("diffusion_steps", 25))
        length_adjust = float(preset_config.get("length_adjust", 1.0))
        cfg_rate = float(preset_config.get("cfg_rate", 0.7))
        model = self.model_cache["default"]

        style, mel_ref, prompt_condition = self._prepare_seed_vc_reference(target_path, cached_embeddings, model)
        source_audios = [
            np.asarray(librosa.load(path, sr=self.sample_rate)[0], dtype=np.float32)
            for path in source_paths
        ]
        chunk_records: list[tuple[int, int, int, np.ndarray]] = []
        generated_wave_chunks: list[list[np.ndarray]] = [[] for _ in source_audios]
        crossfade_samples = int(self.sample_rate * 0.05)

        for item_idx, source_audio in enumerate(source_audios):
            chunks = self.find_silence_boundaries(
                source_audio,
                self.sample_rate,
                min_silence_duration=0.15,
                silence_threshold=0.02,
                max_chunk_duration=25.0,
                min_chunk_duration=3.0,
            )
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                chunk_records.append(
                    (
                        item_idx,
                        chunk_idx,
                        chunk_start,
                        source_audio[chunk_start:chunk_end],
                    )
                )

        for batch_start in range(0, len(chunk_records), max_chunk_batch_size):
            records = chunk_records[batch_start : batch_start + max_chunk_batch_size]
            chunk_audios = [
                torch.from_numpy(record[3]).unsqueeze(0).to(self.device, dtype=torch.float32)
                for record in records
            ]
            chunk_16k = [
                self.torchaudio.functional.resample(chunk_audio, self.sample_rate, 16000)
                for chunk_audio in chunk_audios
            ]
            semantic_batch, _ = self.get_semantic_features_batch(chunk_16k)

            target_lengths = []
            for chunk_audio in chunk_audios:
                mel_chunk = self.to_mel(chunk_audio.float())
                target_lengths.append(int(mel_chunk.size(2) * length_adjust))
            chunk_target_lengths = torch.LongTensor(target_lengths).to(self.device)

            cond_chunk, _, _, _, _ = model.length_regulator(
                semantic_batch,
                ylens=chunk_target_lengths,
                n_quantizers=3,
                f0=None,
            )

            batch_size = len(records)
            prompt_batch = prompt_condition.expand(batch_size, -1, -1)
            mel_ref_batch = mel_ref.expand(batch_size, -1, -1)
            style_batch = style.expand(batch_size, -1)
            cat_condition = torch.cat([prompt_batch, cond_chunk], dim=1)
            x_lens = torch.LongTensor([prompt_condition.size(1) + length for length in target_lengths]).to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.settings.fp16 else torch.float32):
                vc_target = model.cfm.inference(
                    cat_condition,
                    x_lens,
                    mel_ref_batch,
                    style_batch,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=cfg_rate,
                )
                vc_target = vc_target[:, :, mel_ref.size(-1) :]
                vc_wave_batch = self.bigvgan_model(vc_target.float())

            if vc_wave_batch.dim() == 1:
                vc_wave_batch = vc_wave_batch.unsqueeze(0)
            elif vc_wave_batch.dim() == 3:
                vc_wave_batch = vc_wave_batch.squeeze(1)

            for batch_idx, (item_idx, chunk_idx, _, _) in enumerate(records):
                target_samples = max(1, int(target_lengths[batch_idx] * self.hop_length))
                chunk_output = vc_wave_batch[batch_idx, :target_samples].detach().float().cpu().numpy()
                if chunk_idx > 0 and generated_wave_chunks[item_idx] and crossfade_samples > 0:
                    prev_chunk = generated_wave_chunks[item_idx][-1]
                    if len(prev_chunk) >= crossfade_samples and len(chunk_output) >= crossfade_samples:
                        chunk_output = self.seed_vc_crossfade(prev_chunk, chunk_output, crossfade_samples)
                        generated_wave_chunks[item_idx][-1] = prev_chunk[:-crossfade_samples]
                generated_wave_chunks[item_idx].append(chunk_output)

            del chunk_audios, chunk_16k, semantic_batch, chunk_target_lengths, cond_chunk
            del prompt_batch, mel_ref_batch, style_batch, cat_condition, x_lens, vc_target, vc_wave_batch

        results = []
        for chunks in generated_wave_chunks:
            result = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            results.append(self._fade_seed_vc_start(result, self.sample_rate).astype(np.float32))
        return results

    def convert_batch_request(
        self, request: SeedVCBatchRequest, reference_path: Path | None = None,
        embedding_key: str | None = None, cached_embeddings=None,
    ) -> dict[str, Any]:
        items = request.items
        first = items[0]
        for item in items[1:]:
            if (
                item.id != first.id
                or item.style != first.style
                or item.intensity != first.intensity
                or item.preset != first.preset
                or item.reference_url != first.reference_url
            ):
                raise ValueError("Seed-VC batch currently requires a shared id/style/intensity/preset/reference_url")

        preset = first.preset or "default"
        preset_config = self.model_presets.get(preset)
        if preset_config is None:
            raise ValueError(f"Unknown Seed-VC preset {preset!r}; expected one of {sorted(self.model_presets)}")
        chunk_batch_size = min(
            max(1, int(request.max_chunk_batch_size)),
            max(1, int(self.settings.max_chunk_batch_size)),
        )

        import soundfile as sf  # pylint: disable=import-outside-toplevel

        source_paths = [self.tmp_dir / f"{uuid.uuid4().hex}.input" for _ in items]
        wav_output_paths: list[Path] = []
        try:
            for item, source_path in zip(items, source_paths):
                source_path.write_bytes(base64.b64decode(item.audio))

            with self.lock, self.torch.no_grad(), _temporary_cwd(self.root):
                waves = self._convert_voice_v1_batch(
                    source_paths,
                    reference_path,
                    preset_config,
                    cached_embeddings=cached_embeddings,
                    max_chunk_batch_size=chunk_batch_size,
                )
                encoded = []
                for idx, (item, wave_data) in enumerate(zip(items, waves)):
                    wav_output_path = self.output_dir / f"vc_batch_{uuid.uuid4().hex}_{idx}.wav"
                    wav_output_paths.append(wav_output_path)
                    sf.write(wav_output_path, wave_data, self.sample_rate, subtype=MP3_INTERMEDIATE_WAV_SUBTYPE)
                    if item.remove_glitches:
                        self._remove_glitches(wav_output_path)
                    mp3_bytes = _audio_file_to_mp3_bytes(wav_output_path)
                    encoded.append(base64.b64encode(mp3_bytes).decode("ascii"))

            return {
                "sample_rate": self.sample_rate,
                "format": "mp3",
                "preset": preset,
                "count": len(encoded),
                "audios": encoded,
            }
        finally:
            for path in source_paths:
                path.unlink(missing_ok=True)
            for path in wav_output_paths:
                path.unlink(missing_ok=True)

    def _remove_glitches(self, wav_path: Path) -> None:
        if self.process_file is None:
            _LOGGER.warning("Seed-VC glitch removal requested but glitch_remover is unavailable")
            return
        glitch_temp_dir = self.tmp_dir / f"glitch_temp_{uuid.uuid4().hex}"
        glitch_temp_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "rms_win_ms": 5.0,
            "rms_hop_ms": 1.0,
            "rms_thr": 0.002,
            "hold_ms": 15.0,
            "safety_ms": 2.0,
            "max_cut_ms": 200.0,
            "veto_cut_ms": 50.0,
            "fallback_trim_ms": 0.0,
        }
        try:
            result = self.process_file(str(wav_path), str(glitch_temp_dir), params, do_write=True)
            _LOGGER.info("Seed-VC glitch removal: %s cut_ms=%s", result.get("decision"), result.get("cut_ms"))
            shutil.copyfile(result["out"], wav_path)
        finally:
            shutil.rmtree(glitch_temp_dir, ignore_errors=True)

    def convert_request(self, request: SeedVCRequest) -> bytes:
        emb_key, emb = self._resolve_cached_embeddings(request)
        reference_path = None if emb is not None else self._fetch_sample(request)
        return self._convert_with_reference(
            request, reference_path,
            embedding_key=emb_key if emb is not None else None,
            cached_embeddings=emb,
        )

    def convert_generated_audio_batch(
        self,
        source_audios: list[np.ndarray],
        source_sample_rate: int,
        voice_id: str,
        style: str,
        intensity: float,
        preset: str | None,
        output_format: Literal["wav", "mp3"],
        max_chunk_batch_size: int | None = None,
        strict_embedding: bool = False,
    ) -> list[tuple[bytes, float]]:
        if not source_audios:
            return []

        if strict_embedding:
            emb_key, emb = self._resolve_exact_cached_embeddings(voice_id, style, intensity)
        else:
            request = SeedVCRequest(
                audio="",
                id=voice_id,
                style=style,
                intensity=intensity,
                preset=preset,
            )
            emb_key, emb = self._resolve_cached_embeddings(request)
        if emb is None:
            raise ValueError(f"No cached Seed-VC embedding for voice {emb_key!r}")

        preset_name = preset or "default"
        preset_config = self.model_presets.get(preset_name)
        if preset_config is None:
            raise ValueError(f"Unknown Seed-VC preset {preset_name!r}; expected one of {sorted(self.model_presets)}")
        chunk_batch_size = max_chunk_batch_size or self.settings.max_chunk_batch_size

        import soundfile as sf  # pylint: disable=import-outside-toplevel

        source_paths = [self.tmp_dir / f"{uuid.uuid4().hex}.input.wav" for _ in source_audios]
        wav_output_paths: list[Path] = []
        try:
            for source_audio, source_path in zip(source_audios, source_paths):
                sf.write(
                    source_path,
                    _audio_to_float32(source_audio),
                    source_sample_rate,
                    subtype=MP3_INTERMEDIATE_WAV_SUBTYPE,
                )

            _LOGGER.info(
                "Seed-VC convert generated batch: voice=%s preset=%s count=%d",
                voice_id,
                preset_name,
                len(source_paths),
            )
            with self.lock, self.torch.no_grad(), _temporary_cwd(self.root):
                waves = self._convert_voice_v1_batch(
                    source_paths,
                    None,
                    preset_config,
                    cached_embeddings=emb,
                    max_chunk_batch_size=chunk_batch_size,
                )

            encoded: list[tuple[bytes, float]] = []
            for idx, wave_data in enumerate(waves):
                audio_seconds = float(len(wave_data)) / self.sample_rate if self.sample_rate else 0.0
                if output_format == "mp3":
                    wav_output_path = self.output_dir / f"vc_synth_batch_{uuid.uuid4().hex}_{idx}.wav"
                    wav_output_paths.append(wav_output_path)
                    sf.write(wav_output_path, wave_data, self.sample_rate, subtype=MP3_INTERMEDIATE_WAV_SUBTYPE)
                    encoded.append((_audio_file_to_mp3_bytes(wav_output_path), audio_seconds))
                else:
                    encoded.append((_audio_to_wav_bytes(wave_data, self.sample_rate), audio_seconds))
            return encoded
        finally:
            for path in source_paths:
                path.unlink(missing_ok=True)
            for path in wav_output_paths:
                path.unlink(missing_ok=True)

    def convert_generated_audio_reference_batch(
        self,
        source_audios: list[np.ndarray],
        source_sample_rate: int,
        reference_path: Path,
        preset: str | None,
        output_format: Literal["wav", "mp3"],
        max_chunk_batch_size: int | None = None,
    ) -> list[tuple[bytes, float]]:
        if not source_audios:
            return []

        preset_name = preset or "default"
        preset_config = self.model_presets.get(preset_name)
        if preset_config is None:
            raise ValueError(f"Unknown Seed-VC preset {preset_name!r}; expected one of {sorted(self.model_presets)}")
        chunk_batch_size = max_chunk_batch_size or self.settings.max_chunk_batch_size

        import soundfile as sf  # pylint: disable=import-outside-toplevel

        source_paths = [self.tmp_dir / f"{uuid.uuid4().hex}.input.wav" for _ in source_audios]
        wav_output_paths: list[Path] = []
        try:
            for source_audio, source_path in zip(source_audios, source_paths):
                sf.write(
                    source_path,
                    _audio_to_float32(source_audio),
                    source_sample_rate,
                    subtype=MP3_INTERMEDIATE_WAV_SUBTYPE,
                )

            _LOGGER.info(
                "Seed-VC convert generated reference batch: reference=%s preset=%s count=%d",
                reference_path,
                preset_name,
                len(source_paths),
            )
            with self.lock, self.torch.no_grad(), _temporary_cwd(self.root):
                waves = self._convert_voice_v1_batch(
                    source_paths,
                    reference_path,
                    preset_config,
                    cached_embeddings=None,
                    max_chunk_batch_size=chunk_batch_size,
                )

            encoded: list[tuple[bytes, float]] = []
            for idx, wave_data in enumerate(waves):
                audio_seconds = float(len(wave_data)) / self.sample_rate if self.sample_rate else 0.0
                if output_format == "mp3":
                    wav_output_path = self.output_dir / f"vc_synth_sample_batch_{uuid.uuid4().hex}_{idx}.wav"
                    wav_output_paths.append(wav_output_path)
                    sf.write(wav_output_path, wave_data, self.sample_rate, subtype=MP3_INTERMEDIATE_WAV_SUBTYPE)
                    encoded.append((_audio_file_to_mp3_bytes(wav_output_path), audio_seconds))
                else:
                    encoded.append((_audio_to_wav_bytes(wave_data, self.sample_rate), audio_seconds))
            return encoded
        finally:
            for path in source_paths:
                path.unlink(missing_ok=True)
            for path in wav_output_paths:
                path.unlink(missing_ok=True)

    def _convert_with_reference(
        self,
        request: SeedVCRequest,
        reference_path: Path | None,
        embedding_key: str | None = None,
        cached_embeddings: torch.Tensor | None = None,
    ) -> bytes:
        preset = request.preset or "default"
        preset_config = self.model_presets.get(preset)
        if preset_config is None:
            raise ValueError(f"Unknown Seed-VC preset {preset!r}; expected one of {sorted(self.model_presets)}")

        source_path = self.tmp_dir / f"{uuid.uuid4().hex}.input"
        wav_output_path: Path | None = None
        try:
            source_path.write_bytes(base64.b64decode(request.audio))
            _LOGGER.info(
                "Seed-VC convert: voice=%s preset=%s cached_embedding=%s reference=%s",
                request.id, preset, bool(cached_embeddings), reference_path,
            )
            with self.lock:
                wav_output_path = self._convert_voice_v1(
                    source_path,
                    reference_path,
                    preset_config,
                    cached_embeddings=cached_embeddings,
                    voice_id=embedding_key if cached_embeddings is not None else None,
                )
                if request.remove_glitches:
                    self._remove_glitches(wav_output_path)
                return _audio_file_to_mp3_bytes(wav_output_path)
        finally:
            source_path.unlink(missing_ok=True)
            if wav_output_path is not None:
                wav_output_path.unlink(missing_ok=True)

    def find_voice(self, request: SeedVCFindVoiceRequest, reference_path: Path) -> str:
        if self.find_base_voice is None:
            raise RuntimeError("Seed-VC find_voice support is unavailable")
        with self.lock, _temporary_cwd(self.root):
            return str(self.find_base_voice(str(reference_path)))

    def enhance(self, request: SeedVCEnhanceRequest, raw_path: Path) -> bytes:
        import subprocess
        enhance_dir = raw_path.parent
        sample_dir = enhance_dir / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        wav_path = enhance_dir / "sample_raw.wav"

        subprocess.run(["ffmpeg", "-i", str(raw_path), "-t", "120", str(wav_path), "-y"],
                       capture_output=True, check=True)
        subprocess.run(["uv", "tool", "run", "ffmpeg-normalize", str(wav_path), "-o", str(sample_dir / "sample.wav"), "-f"],
                       capture_output=True, check=True)
        sample_wav = sample_dir / "sample.wav"
        enhanced_dir = enhance_dir / "enhanced"
        enhanced_dir.mkdir(exist_ok=True)
        enhanced_wav = enhanced_dir / "sample.wav"
        enhanced_wav.write_bytes(sample_wav.read_bytes())
        mp3_path = enhance_dir / "final_sample.mp3"
        subprocess.run(
            ["ffmpeg", "-i", str(enhanced_wav), "-f", "mp3", "-q:a", "0", "-b:a", "320k", str(mp3_path), "-y"],
            capture_output=True,
            check=True,
        )
        return mp3_path.read_bytes()
