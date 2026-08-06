#!/usr/bin/env python3
"""Generate one continuous VoxCPM utterance and insert forced-aligned pauses."""

from __future__ import annotations

import argparse
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from src.aligned_pauses import insert_aligned_pauses, parse_pause_markers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help='Text containing markers such as "[pause 3s]"')
    parser.add_argument("--output", type=Path, required=True, help="Final WAV output")
    parser.add_argument("--raw-output", type=Path, help="Optional unmodified VoxCPM WAV output")
    parser.add_argument("--report", type=Path, help="Alignment/insertion JSON report")
    parser.add_argument("--server-url", default="http://127.0.0.1:8010")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--voice-id", default="msa.en-GB.OllieMultilingual")
    route.add_argument("--model", choices=("voxcpm",))
    parser.add_argument(
        "--language",
        help="Locale for synthesis/alignment; inferred from --voice-id when omitted",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--alignment-checkpoint",
        type=Path,
        help="Override the production-routed Sparrow alignment checkpoint",
    )
    parser.add_argument("--alignment-config", type=Path)
    parser.add_argument("--alignment-device", default="cuda:2")
    parser.add_argument("--aligner", choices=("ctc", "vits-mas"), default="ctc")
    parser.add_argument(
        "--ctc-model",
        default="MahmoudAshraf/mms-300m-1130-forced-aligner",
    )
    parser.add_argument("--ctc-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--server-config", type=Path, default=Path("local/server.json"))
    parser.add_argument("--voice-catalog", type=Path, default=Path("data/voice-presets.json"))
    return parser.parse_args()


def _normalized_locale(value: str) -> str:
    parts = value.strip().replace("_", "-").split("-")
    if len(parts) == 1:
        return parts[0].lower()
    return "-".join([parts[0].lower(), parts[1].upper(), *parts[2:]])


def _preset_language(voice_id: str, catalog_path: Path) -> str:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    for preset in catalog.get("voices", []):
        if preset.get("id") == voice_id:
            language = str(preset.get("language") or "").strip()
            if not language:
                raise ValueError(f"Voice preset {voice_id!r} has no language")
            return _normalized_locale(language)
    raise ValueError(f"Voice preset {voice_id!r} was not found in {catalog_path}")


def _alignment_checkpoint_for_language(language: str, server_config_path: Path) -> Path:
    """Follow the production Sparrow speaker/model priority for a locale."""
    server_config = json.loads(server_config_path.read_text(encoding="utf-8"))
    piper_config = server_config.get("pipertts") or {}
    aliases = {
        _normalized_locale(str(locale)): str(speaker)
        for locale, speaker in (piper_config.get("lang_speaker_map") or {}).items()
    }
    normalized_language = _normalized_locale(language)
    speaker = aliases.get(normalized_language, normalized_language)
    configured_models = set(piper_config.get("models") or [])
    priorities = piper_config.get("model_priority") or list(configured_models)
    overrides = piper_config.get("model_config") or {}

    for model_name in priorities:
        if configured_models and model_name not in configured_models:
            continue
        override_speakers = (overrides.get(model_name) or {}).get("speakers") or {}
        if override_speakers:
            speakers = override_speakers
        else:
            model_config_path = Path("data") / model_name / "config.json"
            model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
            speakers = model_config.get("speaker_id_map") or {}
        if speaker in speakers:
            checkpoint = Path("data") / model_name / "model.ckpt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Routed alignment checkpoint not found: {checkpoint}")
            return checkpoint

    raise ValueError(
        f"No production Sparrow route for alignment language {language!r} "
        f"(resolved speaker {speaker!r})"
    )


def _resolve_alignment_settings(args: argparse.Namespace) -> tuple[str, Path | None]:
    if args.language:
        language = _normalized_locale(args.language)
    elif args.voice_id:
        language = _preset_language(args.voice_id, args.voice_catalog)
    else:
        raise ValueError("--language is required when synthesizing with --model")
    checkpoint = args.alignment_checkpoint
    if args.aligner == "vits-mas" and checkpoint is None:
        checkpoint = _alignment_checkpoint_for_language(language, args.server_config)
    return language, checkpoint


def _generate_wav(args: argparse.Namespace, text: str) -> tuple[np.ndarray, int, bytes]:
    payload: dict[str, object] = {
        "text": text,
        "format": "wav",
        "language": args.language,
        "seed": args.seed,
    }
    if args.model:
        payload["model"] = args.model
    else:
        payload["voice_id"] = args.voice_id
        # A voice preset chooses its language; specifying both would override
        # catalog routing and is rejected by the public API.
        payload.pop("language")

    request = urllib.request.Request(
        args.server_url.rstrip("/") + "/synthesize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    api_key = os.environ.get("API_KEY", "").strip()
    if api_key:
        request.add_header("X-API-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            wav_bytes = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"VoxCPM request failed with HTTP {exc.code}: {detail}") from exc

    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32), int(sample_rate), wav_bytes


def main() -> int:
    args = _parse_args()
    load_dotenv(".env")
    args.language, alignment_checkpoint = _resolve_alignment_settings(args)
    clean_text, markers = parse_pause_markers(args.text)
    audio, sample_rate, wav_bytes = _generate_wav(args, clean_text)

    if args.aligner == "ctc":
        from src.ctc_forced_alignment import CtcAlignmentConfig, CtcForcedAligner

        validator = CtcForcedAligner(
            CtcAlignmentConfig(
                model=args.ctc_model,
                device=args.alignment_device,
                dtype=args.ctc_dtype,
            )
        )
        alignment = validator.align_words(
            clean_text,
            audio,
            sample_rate,
            language=args.language,
        )
    else:
        from src.duration_alignment import DpBudgetConfig, DurationAlignmentValidator

        assert alignment_checkpoint is not None
        validator = DurationAlignmentValidator(
            DpBudgetConfig(
                checkpoint=alignment_checkpoint,
                config_path=args.alignment_config,
                device=args.alignment_device,
                language=args.language,
                use_bert=True,
                enable_alignment_validation=True,
            )
        )
        alignment = validator.validate_alignment(
            clean_text,
            audio,
            sample_rate,
            language=args.language,
            reject_zero_phoneme_duration=False,
            include_word_timestamps=True,
        )
    word_timestamps = alignment.get("word_timestamps") or []
    final_audio, pause_report = insert_aligned_pauses(
        audio,
        sample_rate,
        markers,
        word_timestamps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, final_audio, sample_rate, subtype="PCM_16")
    raw_output = args.raw_output or args.output.with_name(f"{args.output.stem}.raw.wav")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.write_bytes(wav_bytes)
    report_path = args.report or args.output.with_suffix(".json")
    report = {
        "source_text": args.text,
        "synthesis_text": clean_text,
        "sample_rate": sample_rate,
        "alignment_language": args.language,
        "alignment_backend": args.aligner,
        "alignment_checkpoint": str(alignment_checkpoint) if alignment_checkpoint else None,
        "raw_seconds": audio.size / sample_rate,
        "final_seconds": final_audio.size / sample_rate,
        "alignment": alignment,
        "pauses": pause_report,
        "raw_output": str(raw_output),
        "output": str(args.output),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
