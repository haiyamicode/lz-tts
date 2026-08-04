#!/usr/bin/env python3
"""Confirm GLOBE source accent labels with CommonAccent predictions per speaker."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from src.voxcpm_accent_filter import AccentFilterPolicy


def _path(value: str) -> Path:
    return Path(to_absolute_path(os.path.expanduser(value))).resolve()


def _stable_key(seed: int, speaker_id: str, row: dict[str, Any]) -> bytes:
    value = f"{seed}:{speaker_id}:{row['utterance_id']}:{row['audio']}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if line_number == sum(1 for _ in path.open(encoding="utf-8")):
                    break
                raise
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _aggregate_probabilities(probabilities: list[list[float]]) -> list[float]:
    values = torch.tensor(probabilities, dtype=torch.float64).clamp_min(1e-12)
    mean_log_probability = values.log().mean(dim=0)
    return torch.softmax(mean_log_probability, dim=0).tolist()


@hydra.main(version_base=None, config_path="../local/configs/voxcpm", config_name="globe_commonaccent_relabel")
def main(cfg: DictConfig) -> None:
    # Import lazily so metadata inspection does not require SpeechBrain.
    from speechbrain.inference.interfaces import foreign_class

    dataset_dir = _path(cfg.dataset_dir)
    source_path = dataset_dir / str(cfg.source_manifest)
    output_dir = dataset_dir / str(cfg.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "clip_predictions.jsonl"

    wait_seconds = int(cfg.get("wait_for_source_seconds", 0))
    while not source_path.is_file():
        if wait_seconds <= 0:
            raise FileNotFoundError(source_path)
        print(f"Waiting for source manifest: {source_path}", flush=True)
        time.sleep(wait_seconds)

    rows = _read_jsonl(source_path)
    policy = AccentFilterPolicy.from_mapping(
        OmegaConf.to_container(cfg.accent_filter, resolve=True)
    )
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)

    representatives: list[tuple[str, dict[str, Any]]] = []
    configured_max_clips = cfg.classification.get("max_clips_per_speaker")
    max_clips_per_speaker = (
        int(configured_max_clips) if configured_max_clips is not None else None
    )
    seed = int(cfg.seed)
    for speaker_id, speaker_rows in sorted(by_speaker.items()):
        selected = sorted(speaker_rows, key=lambda row: _stable_key(seed, speaker_id, row))
        if max_clips_per_speaker is not None and max_clips_per_speaker > 0:
            selected = selected[:max_clips_per_speaker]
        representatives.extend((speaker_id, row) for row in selected)

    cached_rows = _read_jsonl(cache_path) if cache_path.is_file() else []
    cached = {str(row["audio"]): row for row in cached_rows}
    pending = [(speaker_id, row) for speaker_id, row in representatives if str(row["audio"]) not in cached]

    if pending:
        wav2vec2_cache_dir = _path(cfg.model.wav2vec2_cache_dir)
        classifier = foreign_class(
            source=str(cfg.model.name),
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": str(cfg.model.device)},
            overrides={"wav2vec2": {"save_path": str(wav2vec2_cache_dir)}},
        )
        label_count = int(classifier.hparams.out_n_neurons)
        labels = list(classifier.hparams.label_encoder.decode_torch(torch.arange(label_count)))
        batch_size = int(cfg.classification.batch_size)
        with cache_path.open("a", encoding="utf-8") as cache_handle:
            progress = tqdm(
                total=len(pending), desc="CommonAccent clips", unit="clip", dynamic_ncols=True
            )
            for start in range(0, len(pending), batch_size):
                batch_items = pending[start : start + batch_size]
                waveforms = [
                    classifier.load_audio(str(row["audio"])).float() for _, row in batch_items
                ]
                lengths = torch.tensor(
                    [waveform.numel() for waveform in waveforms], dtype=torch.float32
                )
                padded = pad_sequence(waveforms, batch_first=True)
                relative_lengths = lengths / float(padded.shape[1])
                with torch.inference_mode():
                    out_prob, scores, indices, predicted_labels = classifier.classify_batch(
                        padded, relative_lengths
                    )
                probabilities = out_prob.detach().cpu().tolist()
                scores = scores.detach().cpu().tolist()
                indices = indices.detach().cpu().tolist()
                for (speaker_id, row), probs, score, index, predicted_label in zip(
                    batch_items, probabilities, scores, indices, predicted_labels
                ):
                    prediction = {
                        "audio": str(row["audio"]),
                        "speaker_id": speaker_id,
                        "utterance_id": str(row["utterance_id"]),
                        "source_locale": str(row["accent"]),
                        "source_accent": str(row["source_accent"]),
                        "classifier_label": str(predicted_label),
                        "classifier_confidence": float(score),
                        "classifier_index": int(index),
                        "probabilities": {
                            label: float(probability)
                            for label, probability in zip(labels, probs)
                        },
                    }
                    cached[prediction["audio"]] = prediction
                    cache_handle.write(
                        json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                cache_handle.flush()
                progress.update(len(batch_items))
            progress.close()
    elif cached:
        labels = list(next(iter(cached.values()))["probabilities"])
    else:
        raise ValueError(f"No cached or pending classifier inputs found in {source_path}")

    speaker_predictions = []
    speaker_lookup = {}
    for speaker_id, speaker_rows in sorted(by_speaker.items()):
        selected = sorted(speaker_rows, key=lambda row: _stable_key(seed, speaker_id, row))
        if max_clips_per_speaker is not None and max_clips_per_speaker > 0:
            selected = selected[:max_clips_per_speaker]
        clip_predictions = [cached[str(row["audio"])] for row in selected]
        aggregate = _aggregate_probabilities(
            [[prediction["probabilities"][label] for label in labels] for prediction in clip_predictions]
        )
        ranked = sorted(range(len(labels)), key=lambda index: aggregate[index], reverse=True)
        label = labels[ranked[0]]
        confidence = float(aggregate[ranked[0]])
        margin = confidence - float(aggregate[ranked[1]])
        clip_labels = [clip_prediction["classifier_label"] for clip_prediction in clip_predictions]
        clip_target_labels = [policy.classifier_target(clip_label) for clip_label in clip_labels]
        source_locales = {str(row["accent"]) for row in speaker_rows}
        source_accents = {str(row["source_accent"]) for row in speaker_rows}
        source_locale = next(iter(source_locales)) if len(source_locales) == 1 else "mixed"
        source_metadata_agrees = policy.source_matches(source_locale, sorted(source_accents))
        classifier_confirms_source = policy.classifier_confirms_speaker(
            source_locale,
            label,
            confidence,
            clip_labels,
        )
        confirming_clips = sum(
            clip_label in policy.classifier_labels.get(source_locale, frozenset())
            for clip_label in clip_labels
        )
        prediction = {
            "speaker_id": speaker_id,
            "source_locales": sorted(source_locales),
            "source_accents": sorted(source_accents),
            "utterances": len(speaker_rows),
            "clips_classified": len(clip_predictions),
            "confirming_clips": confirming_clips,
            "classifier_label": label,
            "classifier_confidence": confidence,
            "classifier_margin": margin,
            "classifier_target_accent": policy.classifier_target(label),
            "clip_labels": clip_labels,
            "clip_target_labels": clip_target_labels,
            "classifier_confirms_source": classifier_confirms_source,
            "source_metadata_agrees": source_metadata_agrees,
            "eligible": source_metadata_agrees and classifier_confirms_source,
        }
        speaker_predictions.append(prediction)
        speaker_lookup[speaker_id] = prediction

    classified = []
    for source_row in rows:
        row = dict(source_row)
        prediction = speaker_lookup[str(row["speaker_id"])]
        sample_prediction = cached[str(row["audio"])]
        sample_label = str(sample_prediction["classifier_label"])
        sample_confidence = float(sample_prediction["classifier_confidence"])
        row["source_locale"] = row["accent"]
        row["accent_classifier_label"] = prediction["classifier_label"]
        row["accent_classifier_target"] = prediction["classifier_target_accent"]
        row["accent_classifier_confidence"] = prediction["classifier_confidence"]
        row["accent_classifier_margin"] = prediction["classifier_margin"]
        row["accent_classifier_confirms_source"] = prediction["classifier_confirms_source"]
        row["accent_source_metadata_agrees"] = prediction["source_metadata_agrees"]
        row["accent_classifier_clips"] = prediction["clips_classified"]
        row["accent_classifier_confirming_clips"] = prediction["confirming_clips"]
        row["accent_sample_classifier_label"] = sample_label
        row["accent_sample_classifier_target"] = policy.classifier_target(sample_label)
        row["accent_sample_classifier_confidence"] = sample_confidence
        row["accent_sample_classifier_passes"] = policy.sample_passes(
            str(row["accent"]), sample_label, sample_confidence
        )
        classified.append(row)

    target_rows = [
        row
        for row in classified
        if row["accent"] in {"en-GB", "en-US"}
        and speaker_lookup[str(row["speaker_id"])]["eligible"]
        and row["accent_sample_classifier_passes"]
    ]
    speaker_gate_rows = [
        row for row in classified if speaker_lookup[str(row["speaker_id"])]["eligible"]
    ]
    _write_jsonl(output_dir / "speaker_predictions.jsonl", speaker_predictions)
    _write_jsonl(output_dir / "classified_all.jsonl", classified)
    _write_jsonl(output_dir / "confirmed_en-GB_en-US.jsonl", target_rows)
    (output_dir / "relabeled_all.jsonl").unlink(missing_ok=True)
    (output_dir / "relabeled_en-GB_en-US.jsonl").unlink(missing_ok=True)

    summary = {
        "source_items": len(rows),
        "speakers": len(speaker_predictions),
        "clips_classified": len(representatives),
        "source_items_by_accent": dict(sorted(Counter(row["accent"] for row in classified).items())),
        "speaker_gate_items": dict(
            sorted(Counter(row["accent"] for row in speaker_gate_rows).items())
        ),
        "sample_gate_items": dict(
            sorted(
                Counter(
                    row["accent"]
                    for row in classified
                    if row["accent_sample_classifier_passes"]
                ).items()
            )
        ),
        "confirmed_items": dict(sorted(Counter(row["accent"] for row in target_rows).items())),
        "confirmed_speakers": dict(
            sorted(
                Counter(
                    accent
                    for _, accent in {
                        (str(row["speaker_id"]), str(row["accent"])) for row in target_rows
                    }
                ).items()
            )
        ),
        "speaker_gate_speakers": dict(
            sorted(
                Counter(
                    prediction["source_locales"][0]
                    for prediction in speaker_predictions
                    if prediction["eligible"] and len(prediction["source_locales"]) == 1
                ).items()
            )
        ),
        "agreement": {
            "scope": "source_label_and_classifier_confirmation",
            "minimum_speaker_confidence": policy.minimum_speaker_confidence,
            "minimum_sample_confidence": policy.minimum_sample_confidence,
            "thresholds_by_accent": {
                accent: {
                    "minimum_speaker_confidence": policy.speaker_confidence_threshold(accent),
                    "minimum_sample_confidence": policy.sample_confidence_threshold(accent),
                }
                for accent in sorted(policy.classifier_labels)
            },
            "minimum_samples_per_speaker": policy.minimum_samples_per_speaker,
            "eligible_speakers": sum(prediction["eligible"] for prediction in speaker_predictions),
        },
        "classifier_target_speakers": dict(
            sorted(Counter(row["classifier_target_accent"] for row in speaker_predictions).items())
        ),
        "classifier_labels_by_speaker": dict(
            sorted(Counter(row["classifier_label"] for row in speaker_predictions).items())
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
