"""
Shared model loading utilities for seed-vc.

This module provides common functions for loading models used across
inference, training, and distillation scripts.
"""

import torch
import torchaudio.compliance.kaldi as kaldi
import yaml

from modules.commons import recursive_munch, build_model, load_checkpoint
from hf_utils import load_custom_model_from_hf


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_dit_model(config_path=None, checkpoint_path=None, device=None):
    """Load DiT model from config and checkpoint.

    Args:
        config_path: Path to config file. If None, uses default whisper-small config.
        checkpoint_path: Path to checkpoint. If None, loads from HuggingFace.
        device: Device to load model on. If None, auto-detects.

    Returns:
        tuple: (model, config, sr, mel_fn_args)
    """
    if device is None:
        device = get_device()

    # Load config and checkpoint
    if config_path is None or checkpoint_path is None:
        hf_checkpoint, hf_config = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
        )
        if config_path is None:
            config_path = hf_config
        if checkpoint_path is None:
            checkpoint_path = hf_checkpoint

    config = yaml.safe_load(open(config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    sr = config["preprocess_params"]["sr"]

    # Build mel function args
    spect_params = config["preprocess_params"]["spect_params"]
    mel_fn_args = {
        "n_fft": spect_params["n_fft"],
        "win_size": spect_params.get("win_length", spect_params.get("win_size", 1024)),
        "hop_size": spect_params.get("hop_length", spect_params.get("hop_size", 256)),
        "num_mels": spect_params.get("n_mels", spect_params.get("num_mels", 80)),
        "sampling_rate": sr,
        "fmin": spect_params.get("fmin", 0),
        "fmax": None if spect_params.get("fmax", "None") == "None" else spect_params["fmax"],
        "center": False,
    }

    # Build model
    model = build_model(model_params, stage="DiT")

    return model, config, sr, mel_fn_args, checkpoint_path


def load_dit_checkpoint(model, checkpoint_path, device=None, is_distill_checkpoint=False):
    """Load checkpoint into DiT model.

    Args:
        model: The model dict (Munch with cfm, length_regulator)
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        is_distill_checkpoint: If True, expects distillation checkpoint format

    Returns:
        tuple: (model, epoch) where epoch is from checkpoint or 0
    """
    if device is None:
        device = get_device()

    if is_distill_checkpoint:
        # Distillation checkpoint format
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get('net', ckpt)
        if 'cfm' in state:
            model.cfm.load_state_dict(state['cfm'], strict=False)
        if 'length_regulator' in state:
            model.length_regulator.load_state_dict(state['length_regulator'], strict=False)
        epoch = ckpt.get('epoch', 0)
    else:
        # Standard HuggingFace/training checkpoint format
        model, _, _, _ = load_checkpoint(
            model, None, checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False
        )
        epoch = 0

    for key in model:
        model[key].eval()
        model[key].to(device)

    return model, epoch


def load_whisper_model(config, device=None):
    """Load Whisper model for semantic feature extraction.

    Args:
        config: Config dict with model_params.speech_tokenizer settings
        device: Device to load on

    Returns:
        tuple: (whisper_model, whisper_feature_extractor)
    """
    if device is None:
        device = get_device()

    from transformers import AutoFeatureExtractor, WhisperModel

    whisper_name = config['model_params']['speech_tokenizer']['name']
    whisper_model = WhisperModel.from_pretrained(whisper_name).to(device)
    del whisper_model.decoder
    whisper_model.eval()
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    return whisper_model, whisper_feature_extractor


def create_semantic_fn(whisper_model, whisper_feature_extractor, device=None):
    """Create semantic feature extraction function.

    Args:
        whisper_model: Loaded Whisper model (encoder only)
        whisper_feature_extractor: Whisper feature extractor
        device: Device for computation

    Returns:
        callable: Function that takes waves_16k tensor and returns semantic features
    """
    if device is None:
        device = get_device()

    def semantic_fn(waves_16k):
        """Extract semantic features from 16kHz audio.

        Args:
            waves_16k: Audio tensor, shape (B, samples) or (samples,)

        Returns:
            Semantic features tensor, shape (B, T, dim)
        """
        # Handle both batched and single inputs
        if waves_16k.dim() == 1:
            waves_16k = waves_16k.unsqueeze(0)

        # Prepare input list for feature extractor
        input_list = [w.cpu().numpy() for w in waves_16k]

        inputs = whisper_feature_extractor(
            input_list,
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        input_features = whisper_model._mask_input_features(
            inputs.input_features, attention_mask=inputs.attention_mask
        ).to(device)

        with torch.no_grad():
            outputs = whisper_model.encoder(
                input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S = outputs.last_hidden_state.to(torch.float32)
        # Trim to match audio length (16kHz / 320 = 50 frames per second)
        S = S[:, :waves_16k.size(-1) // 320 + 1]
        return S

    return semantic_fn


def load_campplus_model(device=None):
    """Load CAMPPlus model for speaker style extraction.

    Args:
        device: Device to load on

    Returns:
        CAMPPlus model in eval mode
    """
    if device is None:
        device = get_device()

    from modules.campplus.DTDNN import CAMPPlus

    campplus_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_path, map_location="cpu", weights_only=False))
    campplus_model.eval().to(device)

    return campplus_model


def extract_style(campplus_model, waves_16k, wave_lengths_16k=None):
    """Extract speaker style embedding from audio.

    Args:
        campplus_model: Loaded CAMPPlus model
        waves_16k: Audio tensor at 16kHz, shape (B, samples)
        wave_lengths_16k: Optional lengths tensor, shape (B,)

    Returns:
        Style embedding tensor, shape (B, 192)
    """
    device = next(campplus_model.parameters()).device
    B = waves_16k.size(0)

    if wave_lengths_16k is None:
        wave_lengths_16k = torch.full((B,), waves_16k.size(-1), device=waves_16k.device)

    feat_list = []
    for bib in range(B):
        feat = kaldi.fbank(
            waves_16k[bib:bib + 1, :wave_lengths_16k[bib]],
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        feat_list.append(feat)

    with torch.no_grad():
        style_list = [campplus_model(f.unsqueeze(0).to(device)) for f in feat_list]
        style = torch.cat(style_list, dim=0)

    return style


def load_bigvgan_vocoder(model_name='nvidia/bigvgan_v2_22khz_80band_256x', device=None):
    """Load BigVGAN vocoder.

    Args:
        model_name: HuggingFace model name
        device: Device to load on

    Returns:
        BigVGAN model in eval mode
    """
    if device is None:
        device = get_device()

    from modules.bigvgan import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(model_name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)

    return bigvgan_model
