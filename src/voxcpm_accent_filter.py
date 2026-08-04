"""Shared source-label and classifier gates for VoxCPM accent datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AccentFilterPolicy:
    minimum_speaker_confidence: float
    minimum_sample_confidence: float
    minimum_samples_per_speaker: int
    classifier_labels: dict[str, frozenset[str]]
    source_accents: dict[str, frozenset[str]]
    thresholds_by_accent: dict[str, tuple[float, float]]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AccentFilterPolicy":
        default_speaker_confidence = float(values["minimum_speaker_confidence"])
        default_sample_confidence = float(values["minimum_sample_confidence"])
        return cls(
            minimum_speaker_confidence=default_speaker_confidence,
            minimum_sample_confidence=default_sample_confidence,
            minimum_samples_per_speaker=int(values["minimum_samples_per_speaker"]),
            classifier_labels={
                str(accent): frozenset(str(label) for label in labels)
                for accent, labels in values["classifier_labels"].items()
            },
            source_accents={
                str(accent): frozenset(str(label) for label in labels)
                for accent, labels in values["source_accents"].items()
            },
            thresholds_by_accent={
                str(accent): (
                    float(
                        thresholds.get(
                            "minimum_speaker_confidence", default_speaker_confidence
                        )
                    ),
                    float(
                        thresholds.get(
                            "minimum_sample_confidence", default_sample_confidence
                        )
                    ),
                )
                for accent, thresholds in values.get("thresholds_by_accent", {}).items()
            },
        )

    def speaker_confidence_threshold(self, accent: str) -> float:
        return self.thresholds_by_accent.get(
            accent,
            (self.minimum_speaker_confidence, self.minimum_sample_confidence),
        )[0]

    def sample_confidence_threshold(self, accent: str) -> float:
        return self.thresholds_by_accent.get(
            accent,
            (self.minimum_speaker_confidence, self.minimum_sample_confidence),
        )[1]

    def classifier_target(self, label: str) -> str:
        matches = [
            accent for accent, labels in self.classifier_labels.items() if label in labels
        ]
        return matches[0] if len(matches) == 1 else "other"

    def source_matches(self, accent: str, source_accents: Sequence[str]) -> bool:
        allowed = self.source_accents.get(accent, frozenset())
        return bool(allowed) and bool(source_accents) and set(source_accents) <= allowed

    def classifier_confirms_speaker(
        self,
        accent: str,
        aggregate_label: str,
        aggregate_confidence: float,
        sample_labels: Sequence[str],
    ) -> bool:
        allowed = self.classifier_labels.get(accent, frozenset())
        return (
            sum(label in allowed for label in sample_labels)
            >= self.minimum_samples_per_speaker
            and aggregate_label in allowed
            and aggregate_confidence >= self.speaker_confidence_threshold(accent)
        )

    def sample_passes(self, accent: str, label: str, confidence: float) -> bool:
        return (
            label in self.classifier_labels.get(accent, frozenset())
            and confidence >= self.sample_confidence_threshold(accent)
        )

    def row_errors(self, row: Mapping[str, Any], expected_accent: str = "") -> list[str]:
        accent = str(row.get("accent", ""))
        errors = []
        if expected_accent and accent != expected_accent:
            errors.append(f"accent {accent!r} != expected {expected_accent!r}")
        if not self.source_matches(accent, [str(row.get("source_accent", ""))]):
            errors.append(f"source accent {row.get('source_accent')!r} is invalid for {accent!r}")
        aggregate_label = str(row.get("accent_classifier_label", ""))
        aggregate_confidence = float(row.get("accent_classifier_confidence", -1.0))
        confirming_samples = int(row.get("accent_classifier_confirming_clips", 0))
        if confirming_samples < self.minimum_samples_per_speaker:
            errors.append(
                f"confirming sample count {confirming_samples} < "
                f"{self.minimum_samples_per_speaker}"
            )
        if aggregate_label not in self.classifier_labels.get(accent, frozenset()):
            errors.append(
                f"aggregate classifier label {aggregate_label!r} is invalid for {accent!r}"
            )
        speaker_confidence_threshold = self.speaker_confidence_threshold(accent)
        if aggregate_confidence < speaker_confidence_threshold:
            errors.append(
                f"aggregate confidence {aggregate_confidence:.6f} < "
                f"{speaker_confidence_threshold:.6f}"
            )
        sample_label = str(row.get("accent_sample_classifier_label", ""))
        sample_confidence = float(row.get("accent_sample_classifier_confidence", -1.0))
        if not self.sample_passes(accent, sample_label, sample_confidence):
            errors.append(
                f"sample classification {sample_label!r}/{sample_confidence:.6f} does not pass"
            )
        return errors

    def require_row(
        self,
        row: Mapping[str, Any],
        *,
        source: str,
        expected_accent: str = "",
    ) -> None:
        errors = self.row_errors(row, expected_accent)
        if errors:
            raise ValueError(f"{source}: {'; '.join(errors)}")
