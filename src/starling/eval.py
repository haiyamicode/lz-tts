from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel

from src.piper.hf_cache import resolve_hf_model_path
from src.piper.preprocess import phonemize_text_for_speaker
from src.piper.semantic import SemanticTokenizer, align_phone_features, build_bert_input
from src.starling import utils
from src.starling.utils.audio import mel_spectrogram, vocos_mel_spectrogram
from src.starling.utils.model import normalize

log = utils.get_pylogger(__name__)


@dataclass
class EvalSample:
    sample_id: str
    kind: str
    text: str
    speaker: str
    speaker_id: int
    phoneme_ids: list[int]
    semantic_features: torch.Tensor
    source_audio_path: str = ""
    source_split: str = ""
    source_index: int | None = None


def run_configured_eval(
    cfg: DictConfig,
    model: torch.nn.Module,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    eval_cfg = cfg.get("eval")
    if not eval_cfg or not bool(eval_cfg.get("enabled", False)):
        return {}

    device_name = str(eval_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device).eval()

    references = list(eval_cfg.get("references") or [])
    if not references:
        raise ValueError("Starling eval requires eval.references")

    samples = _load_dataset_samples(cfg, eval_cfg) + _prepare_ood_samples(cfg, eval_cfg, device)
    if not samples:
        raise ValueError("Starling eval did not select any samples")

    output_dir = _eval_output_dir(cfg, eval_cfg, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = int(eval_cfg.get("sample_rate") or cfg.data.sample_rate)
    reference_cache = _prepare_references(cfg, eval_cfg, references, sample_rate, device)
    vocoder = _load_vocoder(str(eval_cfg.get("vocoder") or "vocos24k"), device)

    rows: list[dict[str, Any]] = []
    wav_paths: list[Path] = []
    with torch.inference_mode():
        for sample in samples:
            for ref_id, ref_data in reference_cache.items():
                wav_path = output_dir / f"{sample.sample_id}__ref_{_safe_id(ref_id)}.wav"
                row = _synthesise_one(
                    cfg=cfg,
                    eval_cfg=eval_cfg,
                    model=model,
                    vocoder=vocoder,
                    sample=sample,
                    ref_id=ref_id,
                    ref_data=ref_data,
                    wav_path=wav_path,
                    device=device,
                )
                rows.append(row)
                wav_paths.append(wav_path)
                log.info("Generated eval sample %s", wav_path)

    if bool((eval_cfg.get("utmos") or {}).get("enabled", False)):
        scores = _score_utmos(wav_paths, eval_cfg.utmos)
        for row in rows:
            row["utmos"] = scores.get(row["wav_path"], "")

    _write_eval_outputs(output_dir, rows, cfg, eval_cfg, checkpoint_path)
    summary = _summary(rows, output_dir)
    log.info("Starling eval complete: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def _eval_output_dir(cfg: DictConfig, eval_cfg: DictConfig, checkpoint_path: str | Path | None) -> Path:
    root = Path(str(eval_cfg.get("output_dir") or Path(cfg.paths.output_dir) / "eval"))
    if checkpoint_path:
        checkpoint_name = Path(checkpoint_path).stem.replace("=", "")
    else:
        checkpoint_name = "current"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"{checkpoint_name}_{stamp}"


def _load_dataset_samples(cfg: DictConfig, eval_cfg: DictConfig) -> list[EvalSample]:
    sample_cfgs = list(eval_cfg.get("dataset_samples") or [])
    samples: list[EvalSample] = []
    for sample_cfg in sample_cfgs:
        split = str(sample_cfg.get("split", "train"))
        filelist_path = Path(str(cfg.data.train_filelist_path if split == "train" else cfg.data.valid_filelist_path))
        language = sample_cfg.get("language")
        count = int(sample_cfg.get("count", 0))
        if count <= 0:
            continue
        selected = _select_jsonl_rows(filelist_path, count=count, language=language)
        for row_index, row in selected:
            text = str(row.get("text") or "")
            speaker = str(row.get("speaker") or language or "")
            speaker_id = int(row["speaker_id"])
            semantic_features = torch.load(row["bert_path"], map_location="cpu", weights_only=True).float()
            phoneme_ids = [int(item) for item in row["phoneme_ids"]]
            if semantic_features.shape[-1] != len(phoneme_ids):
                raise ValueError(
                    f"BERT/text length mismatch in {filelist_path}:{row_index}: "
                    f"bert_len={semantic_features.shape[-1]} text_len={len(phoneme_ids)}"
                )
            samples.append(
                EvalSample(
                    sample_id=f"{split}_{row_index:05d}_{_safe_id(speaker or 'spk')}",
                    kind=f"{split}_dataset",
                    text=text,
                    speaker=speaker,
                    speaker_id=speaker_id,
                    phoneme_ids=phoneme_ids,
                    semantic_features=semantic_features,
                    source_audio_path=str(row.get("audio_path") or ""),
                    source_split=split,
                    source_index=row_index,
                )
            )
    return samples


def _select_jsonl_rows(path: Path, count: int, language: str | None = None) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip():
                continue
            row = json.loads(line)
            if language and str(row.get("speaker") or "").lower() != str(language).lower():
                continue
            selected.append((index, row))
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise ValueError(f"Only selected {len(selected)} rows from {path}, requested {count}")
    return selected


def _prepare_ood_samples(cfg: DictConfig, eval_cfg: DictConfig, device: torch.device) -> list[EvalSample]:
    ood_cfgs = list(eval_cfg.get("ood_samples") or [])
    if not ood_cfgs:
        return []

    preprocess_cfg = eval_cfg.get("preprocess") or {}
    piper_config = Path(str(preprocess_cfg.get("piper_config")))
    if not piper_config:
        raise ValueError("eval.preprocess.piper_config is required for OOD samples")
    semantic_model_name = str(preprocess_cfg.get("semantic_model_name") or "distilbert/distilbert-base-multilingual-cased")
    semantic_max_tokens = preprocess_cfg.get("semantic_max_tokens")
    semantic_max_tokens = None if semantic_max_tokens is None else int(semantic_max_tokens)

    tokenizer = SemanticTokenizer(semantic_model_name, max_length=semantic_max_tokens)
    semantic_path = resolve_hf_model_path(semantic_model_name, require_weights=True)
    semantic_model = AutoModel.from_pretrained(semantic_path).to(device).eval()

    samples: list[EvalSample] = []
    for index, sample_cfg in enumerate(ood_cfgs):
        text = str(sample_cfg["text"])
        speaker = str(sample_cfg.get("speaker") or sample_cfg.get("language") or "en")
        row = phonemize_text_for_speaker(
            text,
            piper_config,
            speaker_label=speaker,
            neural=bool(preprocess_cfg.get("neural", True)),
        )
        phoneme_ids = [int(item) for item in row["phoneme_ids"]]
        bert_input = build_bert_input(
            [str(row["text"])],
            tokenizer,
            phoneme_lengths=[len(phoneme_ids)],
            word_spans=[row.get("word_spans")],
        )
        if bert_input is None or "word2ph" not in bert_input:
            raise RuntimeError(f"Failed to build semantic input for OOD eval sample {index}")
        with torch.inference_mode():
            hidden = semantic_model(
                input_ids=bert_input["input_ids"].to(device),
                attention_mask=bert_input["attention_mask"].to(device),
            ).last_hidden_state[0]
        semantic_features = align_phone_features(
            hidden,
            bert_input["word2ph"][0].to(device),
            len(phoneme_ids),
        ).detach().float().cpu()
        samples.append(
            EvalSample(
                sample_id=str(sample_cfg.get("id") or f"ood_{index:03d}_{_safe_id(speaker)}"),
                kind="ood_text",
                text=str(row["text"]),
                speaker=speaker,
                speaker_id=int(row["speaker_id"]),
                phoneme_ids=phoneme_ids,
                semantic_features=semantic_features,
            )
        )
    return samples


def _prepare_references(
    cfg: DictConfig,
    eval_cfg: DictConfig,
    references: list[DictConfig],
    sample_rate: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    seed_vc_root = Path(str(cfg.paths.root_dir)) / "src" / "seed_vc_runtime"
    if str(seed_vc_root) not in sys.path:
        sys.path.insert(0, str(seed_vc_root))
    from hf_utils import load_custom_model_from_hf  # pylint: disable=import-outside-toplevel
    from modules.campplus.DTDNN import CAMPPlus  # pylint: disable=import-outside-toplevel

    checkpoint_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus = CAMPPlus(feat_dim=80, embedding_size=int(eval_cfg.get("prompt_embedding_dim") or cfg.data.prompt_embedding_dim))
    campplus.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    campplus.eval().to(device)

    out: dict[str, dict[str, Any]] = {}
    for ref in references:
        ref_id = str(ref.get("id") or Path(str(ref["path"])).stem)
        path = Path(str(ref["path"]))
        audio = _load_audio(path, sample_rate)
        prompt_mel = _reference_prompt_mel(cfg, audio, sample_rate)
        original_frames = int(prompt_mel.shape[-1])
        max_frames = int(eval_cfg.get("prompt_mel_max_frames") or cfg.data.prompt_mel_max_frames or 0)
        if max_frames > 0 and prompt_mel.shape[-1] > max_frames:
            prompt_mel = prompt_mel[:, :max_frames]
        prompt_embedding = _extract_campplus(campplus, audio, sample_rate, device).cpu()
        out[ref_id] = {
            "path": str(path),
            "audio_seconds": float(audio.shape[-1] / sample_rate),
            "prompt_mel": prompt_mel.cpu(),
            "prompt_mel_length": int(prompt_mel.shape[-1]),
            "prompt_original_frames": original_frames,
            "prompt_embedding": prompt_embedding,
        }
    return out


def _load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    if path.suffix == ".pt":
        audio = torch.load(path, map_location="cpu", weights_only=True).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio

    audio_np, source_rate = sf.read(path, always_2d=True, dtype="float32")
    audio = torch.from_numpy(audio_np.T.copy())
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio.contiguous()


def _extract_campplus(campplus: torch.nn.Module, audio: torch.Tensor, sample_rate: int, device: torch.device) -> torch.Tensor:
    audio_16k = torchaudio.functional.resample(audio, sample_rate, 16000)
    feat = kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = campplus(feat.unsqueeze(0).to(device)).squeeze(0).detach().float()
    if embedding.shape != (192,):
        raise ValueError(f"Expected CAMP++ embedding shape (192,), got {tuple(embedding.shape)}")
    return embedding


def _reference_prompt_mel(cfg: DictConfig, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if cfg.data.mel_backend == "vocos_mel_24khz":
        mel = vocos_mel_spectrogram(
            audio,
            sampling_rate=sample_rate,
            n_fft=int(cfg.data.n_fft),
            hop_size=int(cfg.data.hop_length),
            num_mels=int(cfg.data.n_feats),
        ).squeeze()
    else:
        mel = mel_spectrogram(
            audio,
            int(cfg.data.n_fft),
            int(cfg.data.n_feats),
            sample_rate,
            int(cfg.data.hop_length),
            int(cfg.data.win_length),
            cfg.data.f_min,
            cfg.data.f_max,
            center=False,
        ).squeeze()
    return normalize(mel, cfg.data.data_statistics.mel_mean, cfg.data.data_statistics.mel_std)


def _synthesise_one(
    *,
    cfg: DictConfig,
    eval_cfg: DictConfig,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    sample: EvalSample,
    ref_id: str,
    ref_data: dict[str, Any],
    wav_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    x = torch.tensor(sample.phoneme_ids, dtype=torch.long, device=device).unsqueeze(0)
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    spks = torch.tensor([sample.speaker_id], dtype=torch.long, device=device)
    semantic_features = sample.semantic_features.unsqueeze(0).to(device)
    prompt_mel = ref_data["prompt_mel"].unsqueeze(0).to(device)
    prompt_mel_lengths = torch.tensor([int(ref_data["prompt_mel_length"])], dtype=torch.long, device=device)
    prompt_embedding = ref_data["prompt_embedding"].unsqueeze(0).to(device)

    output = model.synthesise(
        x,
        x_lengths,
        n_timesteps=int(eval_cfg.get("n_timesteps", 32)),
        temperature=float(eval_cfg.get("temperature", 1.0)),
        spks=spks,
        length_scale=float(eval_cfg.get("length_scale", 1.0)),
        semantic_features=semantic_features,
        noise_scale_w=eval_cfg.get("noise_scale_w"),
        sdp_ratio=eval_cfg.get("sdp_ratio"),
        prompt_mel=prompt_mel,
        prompt_mel_lengths=prompt_mel_lengths,
        prompt_embedding=prompt_embedding,
    )
    audio = vocoder.decode(output["mel"]).clamp(-1, 1).squeeze().detach().float().cpu().numpy()
    sf.write(wav_path, np.clip(audio, -1.0, 1.0), int(eval_cfg.get("sample_rate") or cfg.data.sample_rate), subtype="PCM_24")

    return {
        "sample_id": sample.sample_id,
        "sample_kind": sample.kind,
        "source_split": sample.source_split,
        "source_index": "" if sample.source_index is None else sample.source_index,
        "speaker": sample.speaker,
        "speaker_id": sample.speaker_id,
        "reference_id": ref_id,
        "reference_path": ref_data["path"],
        "reference_seconds": ref_data["audio_seconds"],
        "prompt_original_frames": ref_data["prompt_original_frames"],
        "prompt_used_frames": ref_data["prompt_mel_length"],
        "text": sample.text,
        "source_audio_path": sample.source_audio_path,
        "wav_path": str(wav_path),
        "duration_sec": float(len(audio) / int(eval_cfg.get("sample_rate") or cfg.data.sample_rate)),
        "mel_frames": int(output["mel"].shape[-1]),
        "mel_length": int(output["mel_lengths"][0].item()),
        "rtf_model": float(output["rtf"]),
        "utmos": "",
    }


def _load_vocoder(name: str, device: torch.device) -> torch.nn.Module:
    if name != "vocos24k":
        raise ValueError(f"Unsupported Starling eval vocoder: {name}")
    from vocos import Vocos  # pylint: disable=import-outside-toplevel

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="vocos.pretrained")
        return Vocos.from_pretrained("charactr/vocos-mel-24khz").eval().to(device)


def _score_utmos(wav_paths: list[Path], utmos_cfg: DictConfig) -> dict[str, float]:
    python_bin = Path(str(utmos_cfg["python_bin"]))
    worker_path = Path(str(utmos_cfg["worker_path"]))
    if not python_bin.exists() or not worker_path.exists():
        raise RuntimeError(f"UTMOS worker is missing: python={python_bin} worker={worker_path}")

    env = os.environ.copy()
    if utmos_cfg.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(utmos_cfg.cuda_visible_devices)

    proc = subprocess.Popen(
        [str(python_bin), str(worker_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    scores: dict[str, float] = {}
    try:
        ready_line = proc.stdout.readline()
        if not ready_line:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"UTMOS worker exited before ready: {stderr.strip()}")
        for idx, wav_path in enumerate(wav_paths):
            proc.stdin.write(json.dumps({"id": idx, "path": str(wav_path)}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"UTMOS worker stopped before scoring {wav_path}: {stderr.strip()}")
            response = json.loads(line)
            if "mos_score" not in response:
                raise RuntimeError(f"UTMOS scoring failed for {wav_path}: {response}")
            scores[str(wav_path)] = float(response["mos_score"])
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    return scores


def _write_eval_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    cfg: DictConfig,
    eval_cfg: DictConfig,
    checkpoint_path: str | Path | None,
) -> None:
    fieldnames = list(rows[0].keys())
    with (output_dir / "eval_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "eval_scores.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "config_name": cfg.get("run_name"),
        "eval": OmegaConf.to_container(eval_cfg, resolve=True),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def _summary(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    values = [float(row["utmos"]) for row in rows if row.get("utmos") not in ("", None)]
    by_kind: dict[str, list[float]] = {}
    by_ref: dict[str, list[float]] = {}
    for row in rows:
        if row.get("utmos") in ("", None):
            continue
        score = float(row["utmos"])
        by_kind.setdefault(str(row["sample_kind"]), []).append(score)
        by_ref.setdefault(str(row["reference_id"]), []).append(score)
    return {
        "output_dir": str(output_dir),
        "num_wavs": len(rows),
        "utmos_mean": float(np.mean(values)) if values else None,
        "utmos_by_kind": {key: float(np.mean(val)) for key, val in by_kind.items()},
        "utmos_by_reference": {key: float(np.mean(val)) for key, val in by_ref.items()},
        "scores_csv": str(output_dir / "eval_scores.csv"),
    }


def _safe_id(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
    safe = "_".join(part for part in safe.split("_") if part)
    if not safe:
        safe = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return safe[:80]
