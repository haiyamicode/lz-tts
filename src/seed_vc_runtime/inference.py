import os

import numpy as np

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import shutil
import warnings
import argparse
import torch
import yaml

warnings.simplefilter('ignore')

# load packages
import random

from modules.commons import *
import time

import torchaudio
import librosa
from modules.commons import str2bool

from hf_utils import load_custom_model_from_hf


# Load model and configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

fp16 = False
def load_models(args):
    global fp16
    fp16 = args.fp16
    if not args.f0_condition:
        if args.checkpoint is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                                                                            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
        else:
            dit_checkpoint_path = args.checkpoint
            dit_config_path = args.config
        f0_fn = None
    else:
        if args.checkpoint is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                             "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                                                                             "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
        else:
            dit_checkpoint_path = args.checkpoint
            dit_config_path = args.config
        # f0 extractor
        from modules.rmvpe import RMVPE

        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        f0_extractor = RMVPE(model_path, is_half=False, device=device)
        f0_fn = f0_extractor.infer_from_audio

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu", weights_only=False))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu', weights_only=False))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  padding=True,
                                                  sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
        sr,
        hop_length,
    )

def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def find_silence_boundaries(audio, sr, min_silence_duration=0.1, silence_threshold=0.01,
                            max_chunk_duration=25.0, min_chunk_duration=5.0):
    """
    Find optimal chunk boundaries at silence points in audio.

    Args:
        audio: numpy array of audio samples
        sr: sample rate
        min_silence_duration: minimum duration of silence to consider (seconds)
        silence_threshold: RMS threshold below which audio is considered silent
        max_chunk_duration: maximum chunk duration (seconds)
        min_chunk_duration: minimum chunk duration (seconds)

    Returns:
        List of (start_sample, end_sample) tuples for each chunk
    """
    # Calculate RMS energy in small windows
    window_size = int(sr * 0.02)  # 20ms windows
    hop_size = int(sr * 0.01)     # 10ms hop

    # Compute RMS for each window
    num_windows = (len(audio) - window_size) // hop_size + 1
    rms = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * hop_size
        window = audio[start:start + window_size]
        rms[i] = np.sqrt(np.mean(window ** 2))

    # Normalize RMS
    max_rms = np.max(rms) if np.max(rms) > 0 else 1.0
    rms_normalized = rms / max_rms

    # Find silence regions (consecutive windows below threshold)
    is_silent = rms_normalized < silence_threshold
    min_silent_windows = int(min_silence_duration / 0.01)  # Convert to window count

    # Find silence boundaries (transitions from speech to silence)
    silence_points = []  # (sample_position, silence_duration_samples)

    i = 0
    while i < len(is_silent):
        if is_silent[i]:
            # Found start of silence
            silence_start = i
            while i < len(is_silent) and is_silent[i]:
                i += 1
            silence_end = i
            silence_duration = silence_end - silence_start

            if silence_duration >= min_silent_windows:
                # Use the middle of the silence region as the cut point
                mid_window = (silence_start + silence_end) // 2
                sample_pos = mid_window * hop_size
                silence_points.append((sample_pos, silence_duration * hop_size))
        else:
            i += 1

    # Now create chunks based on silence points
    # Only split when approaching max_chunk_duration
    max_chunk_samples = int(max_chunk_duration * sr)

    chunks = []
    chunk_start = 0
    last_good_silence = None  # Track the last silence point before exceeding max

    for silence_pos, _ in silence_points:
        chunk_length = silence_pos - chunk_start

        if chunk_length >= max_chunk_samples:
            # We've exceeded max - cut at the last good silence point, or here if none
            if last_good_silence is not None and last_good_silence > chunk_start:
                chunks.append((chunk_start, last_good_silence))
                chunk_start = last_good_silence
                last_good_silence = silence_pos
            else:
                # No good silence point found, cut here
                chunks.append((chunk_start, silence_pos))
                chunk_start = silence_pos
                last_good_silence = None
        else:
            # Remember this as a potential cut point
            last_good_silence = silence_pos

    # Add final chunk
    if chunk_start < len(audio):
        chunks.append((chunk_start, len(audio)))

    # If no chunks were created, fall back to single chunk
    if not chunks:
        chunks = [(0, len(audio))]

    return chunks


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@torch.no_grad()
def convert_voice(
    source_audio_path: str,
    ref_audio_path: str,
    model,
    semantic_fn,
    vocoder_fn,
    campplus_model,
    mel_fn,
    sr: int = 22050,
    hop_length: int = 256,
    diffusion_steps: int = 30,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.7,
    f0_condition: bool = False,
    f0_fn=None,
    auto_f0_adjust: bool = False,
    pitch_shift: int = 0,
    device=None,
    fp16: bool = True,
    max_chunk_duration: float = 25.0,
    cached_ref_embeddings: dict = None,
):
    """
    Core voice conversion function with silence-based chunking.
    Can be called from both inference.py CLI and server.py API.

    Args:
        source_audio_path: Path to source audio file
        ref_audio_path: Path to reference/target voice audio file (can be None if cached_ref_embeddings provided)
        model: The DiT model dict with 'cfm' and 'length_regulator'
        semantic_fn: Function to extract semantic features (e.g., Whisper)
        vocoder_fn: Vocoder model (e.g., BigVGAN)
        campplus_model: Speaker encoder model (can be None if cached_ref_embeddings provided)
        mel_fn: Mel spectrogram function (can be None if cached_ref_embeddings provided)
        sr: Sample rate (22050 for non-f0, 44100 for f0)
        hop_length: Hop length for mel spectrogram
        diffusion_steps: Number of diffusion steps
        length_adjust: Length adjustment factor
        inference_cfg_rate: CFG rate for inference
        f0_condition: Whether to use F0 conditioning
        f0_fn: F0 extraction function (required if f0_condition=True)
        auto_f0_adjust: Whether to auto-adjust F0
        pitch_shift: Pitch shift in semitones
        device: Torch device
        fp16: Whether to use FP16
        max_chunk_duration: Maximum duration per chunk in seconds
        cached_ref_embeddings: Optional dict with pre-computed reference embeddings
            Keys: 'style', 'mel_ref', 'prompt_condition'

    Returns:
        numpy array of converted audio at sample rate sr
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load source audio
    source_audio = librosa.load(source_audio_path, sr=sr)[0]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    # Get reference embeddings (from cache or compute from audio)
    if cached_ref_embeddings is not None:
        # Use pre-computed embeddings
        style = cached_ref_embeddings['style'].to(device)
        mel_ref = cached_ref_embeddings['mel_ref'].to(device)
        prompt_condition = cached_ref_embeddings['prompt_condition'].to(device)
        print("Using cached reference embeddings")
    else:
        # Compute from reference audio
        ref_audio = librosa.load(ref_audio_path, sr=sr)[0]
        ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

        # Reference audio processing
        ref_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        S_ref = semantic_fn(ref_16k)
        mel_ref = mel_fn(ref_audio.to(device).float())
        ref_lengths = torch.LongTensor([mel_ref.size(2)]).to(device)

        # Speaker embedding
        feat_ref = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat_ref = feat_ref - feat_ref.mean(dim=0, keepdim=True)
        style = campplus_model(feat_ref.unsqueeze(0))

        # F0 for reference if needed
        F0_ref = None
        if f0_condition and f0_fn is not None:
            F0_ref = f0_fn(ref_16k[0], thred=0.03)
            F0_ref = torch.from_numpy(F0_ref).to(device)[None]

        # Reference prompt condition
        prompt_condition, _, _, _, _ = model.length_regulator(S_ref, ylens=ref_lengths, n_quantizers=3, f0=F0_ref)

    # Find silence-based chunk boundaries in source audio
    source_audio_np = source_audio.squeeze(0).cpu().numpy()
    audio_chunks = find_silence_boundaries(
        source_audio_np, sr,
        min_silence_duration=0.15,
        silence_threshold=0.02,
        max_chunk_duration=max_chunk_duration,
        min_chunk_duration=3.0
    )

    print(f"Split audio into {len(audio_chunks)} chunks at silence boundaries")

    # Process each chunk independently
    generated_wave_chunks = []
    crossfade_samples = int(sr * 0.05)  # 50ms crossfade

    for chunk_idx, (chunk_start, chunk_end) in enumerate(audio_chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(audio_chunks)}: {chunk_start/sr:.2f}s - {chunk_end/sr:.2f}s")

        # Extract chunk audio
        chunk_audio = source_audio[:, chunk_start:chunk_end]
        chunk_16k = torchaudio.functional.resample(chunk_audio, sr, 16000)

        # Get semantic features for this chunk
        if chunk_16k.size(-1) <= 16000 * 30:
            S_chunk = semantic_fn(chunk_16k)
        else:
            # For very long chunks, use overlapping windows for whisper
            overlapping_time = 5
            S_chunk_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < chunk_16k.size(-1):
                if buffer is None:
                    window = chunk_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    window = torch.cat(
                        [buffer, chunk_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                        dim=-1)
                S_window = semantic_fn(window)
                if traversed_time == 0:
                    S_chunk_list.append(S_window)
                else:
                    S_chunk_list.append(S_window[:, 50 * overlapping_time:])
                buffer = window[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else window.size(-1) - 16000 * overlapping_time
            S_chunk = torch.cat(S_chunk_list, dim=1)

        # Mel and length for chunk
        mel_chunk = mel_fn(chunk_audio.to(device).float())
        chunk_target_lengths = torch.LongTensor([int(mel_chunk.size(2) * length_adjust)]).to(device)

        # F0 for chunk if needed
        chunk_f0 = None
        if f0_condition and f0_fn is not None:
            F0_chunk = f0_fn(chunk_16k[0].cpu(), thred=0.03)
            F0_chunk = torch.from_numpy(F0_chunk).to(device)[None]

            if auto_f0_adjust:
                # Get reference F0 stats
                voiced_F0_ref = F0_ref[F0_ref > 1] if F0_ref is not None else None
                voiced_F0_chunk = F0_chunk[F0_chunk > 1]
                if voiced_F0_ref is not None and len(voiced_F0_ref) > 0 and len(voiced_F0_chunk) > 0:
                    log_f0_chunk = torch.log(F0_chunk + 1e-5)
                    median_log_f0_ref = torch.median(torch.log(voiced_F0_ref + 1e-5))
                    median_log_f0_chunk = torch.median(torch.log(voiced_F0_chunk + 1e-5))
                    log_f0_chunk[F0_chunk > 1] = log_f0_chunk[F0_chunk > 1] - median_log_f0_chunk + median_log_f0_ref
                    F0_chunk = torch.exp(log_f0_chunk)

            if pitch_shift != 0:
                F0_chunk[F0_chunk > 1] = adjust_f0_semitones(F0_chunk[F0_chunk > 1], pitch_shift)

            chunk_f0 = F0_chunk

        # Length regulation for source chunk
        cond_chunk, _, _, _, _ = model.length_regulator(S_chunk, ylens=chunk_target_lengths, n_quantizers=3, f0=chunk_f0)

        # Concatenate prompt condition with chunk condition
        cat_condition = torch.cat([prompt_condition, cond_chunk], dim=1)

        # CFM inference
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(device),
                mel_ref, style, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate
            )
            vc_target = vc_target[:, :, mel_ref.size(-1):]

        # Vocoder
        vc_wave_chunk = vocoder_fn(vc_target.float()).squeeze()
        chunk_output = vc_wave_chunk.cpu().numpy()

        # Apply crossfade with previous chunk (at silence boundary)
        if chunk_idx > 0 and len(generated_wave_chunks) > 0 and crossfade_samples > 0:
            prev_chunk = generated_wave_chunks[-1]
            if len(prev_chunk) >= crossfade_samples and len(chunk_output) >= crossfade_samples:
                chunk_output = crossfade(prev_chunk, chunk_output, crossfade_samples)
                generated_wave_chunks[-1] = prev_chunk[:-crossfade_samples]

        generated_wave_chunks.append(chunk_output)

    # Concatenate all chunks
    result = np.concatenate(generated_wave_chunks)

    # Detect if there's a startup transient vs actual speech content
    # Transient pattern: high energy in first 10-15ms that drops off
    # Speech pattern: sustained or rising energy
    check_window = int(sr * 0.03)  # Check first 30ms
    if len(result) > check_window:
        # Compare energy in first 10ms vs 20-30ms
        first_10ms = int(sr * 0.01)
        window_20_30ms = result[int(sr * 0.02):check_window]

        energy_first = np.sqrt(np.mean(result[:first_10ms] ** 2))
        energy_later = np.sqrt(np.mean(window_20_30ms ** 2))

        # If first 10ms has higher energy than 20-30ms, likely a transient
        # Apply aggressive fade. Otherwise, use gentle fade to preserve content.
        if energy_first > energy_later * 1.5:
            # Transient detected - use aggressive x^4 fade
            fade_samples = int(sr * 0.025)
            t = np.linspace(0, 1, fade_samples)
            fade_in = t ** 4
            result[:fade_samples] *= fade_in
        else:
            # Actual content at start - use gentle linear fade (just prevent DC pop)
            fade_samples = int(sr * 0.005)  # 5ms gentle fade
            fade_in = np.linspace(0, 1, fade_samples)
            result[:fade_samples] *= fade_in

    return result


@torch.no_grad()
def main(args):
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args, sr, hop_length = load_models(args)

    time_vc_start = time.time()

    # Use the shared convert_voice function
    vc_wave = convert_voice(
        source_audio_path=args.source,
        ref_audio_path=args.target,
        model=model,
        semantic_fn=semantic_fn,
        vocoder_fn=vocoder_fn,
        campplus_model=campplus_model,
        mel_fn=mel_fn,
        sr=sr,
        hop_length=hop_length,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        inference_cfg_rate=args.inference_cfg_rate,
        f0_condition=args.f0_condition,
        f0_fn=f0_fn,
        auto_f0_adjust=args.auto_f0_adjust,
        pitch_shift=args.semi_tone_shift,
        device=device,
        fp16=fp16,
    )

    time_vc_end = time.time()
    print(f"RTF: {(time_vc_end - time_vc_start) / len(vc_wave) * sr}")

    source_name = os.path.basename(args.source).split(".")[0]
    target_name = os.path.basename(args.target).split(".")[0]
    os.makedirs(args.output, exist_ok=True)
    output_file_path = os.path.join(args.output, f"vc_{source_name}_{target_name}_{args.length_adjust}_{args.diffusion_steps}_{args.inference_cfg_rate}.wav")
    torchaudio.save(output_file_path, torch.tensor(vc_wave)[None, :], sr)
    return output_file_path
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./examples/source/source_s1.wav")
    parser.add_argument("--target", type=str, default="./examples/reference/s1p1.wav")
    parser.add_argument("--output", type=str, default="./reconstructed")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", type=str2bool, default=False)
    parser.add_argument("--auto-f0-adjust", type=str2bool, default=False)
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file", default=None)
    parser.add_argument("--config", type=str, help="Path to the config file", default=None)
    parser.add_argument("--fp16", type=str2bool, default=True)
    args = parser.parse_args()
    main(args)
