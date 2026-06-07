import json
import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset

_LOGGER = logging.getLogger("vits.dataset")
_DEBUG_SEMANTIC = bool(int(os.environ.get("PIPER_SEMANTIC_DEBUG", "0")))


@dataclass
class Utterance:
    phoneme_ids: List[int]
    audio_norm_path: Path
    audio_spec_path: Path
    audio_path: Optional[Path] = None
    speaker_id: Optional[int] = None
    text: Optional[str] = None
    word_spans: Optional[List[List[int]]] = None
    bert_path: Optional[Path] = None
    bert_dim: Optional[int] = None
    spec_length: Optional[int] = None


@dataclass
class UtteranceTensors:
    phoneme_ids: LongTensor
    spectrogram: FloatTensor
    audio_norm: FloatTensor
    speaker_id: Optional[LongTensor] = None
    text: Optional[str] = None
    word_spans: Optional[List[List[int]]] = None
    bert_features: Optional[FloatTensor] = None

    @property
    def spec_length(self) -> int:
        return self.spectrogram.size(1)


@dataclass
class Batch:
    phoneme_ids: LongTensor
    phoneme_lengths: LongTensor
    spectrograms: FloatTensor
    spectrogram_lengths: LongTensor
    audios: FloatTensor
    audio_lengths: LongTensor
    speaker_ids: Optional[LongTensor] = None
    texts: Optional[List[str]] = None
    word_spans: Optional[List[Optional[List[List[int]]]]] = None
    bert_features: Optional[FloatTensor] = None


class PiperDataset(Dataset):
    """
    Dataset format:

    * phoneme_ids (required)
    * audio_norm_path (required)
    * audio_spec_path (required)
    * text (optional)
    * phonemes (optional)
    * audio_path (optional)
    """

    def __init__(
        self,
        dataset_paths: List[Union[str, Path]],
        max_phoneme_ids: Optional[int] = None,
        hop_length: int = 256,
    ):
        self.utterances: List[Utterance] = []
        self.lengths: List[int] = []
        self.hop_length = int(hop_length)

        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            _LOGGER.debug("Loading dataset: %s", dataset_path)
            self.utterances.extend(
                PiperDataset.load_dataset(
                    dataset_path,
                    max_phoneme_ids=max_phoneme_ids,
                    hop_length=self.hop_length,
                )
            )
        self.lengths = [
            int(utt.spec_length or max(1, len(utt.phoneme_ids)))
            for utt in self.utterances
        ]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx) -> UtteranceTensors:
        utt = self.utterances[idx]
        bert_features = None
        if utt.bert_path is not None:
            loaded_features = torch.load(utt.bert_path, map_location="cpu")
            if isinstance(loaded_features, dict):
                loaded_features = loaded_features["features"]
            bert_features = loaded_features.float()
            if bert_features.dim() != 2:
                raise ValueError(
                    f"Expected 2D BERT features at {utt.bert_path}, got {tuple(bert_features.shape)}"
                )

        return UtteranceTensors(
            phoneme_ids=LongTensor(utt.phoneme_ids),
            audio_norm=torch.load(utt.audio_norm_path),
            spectrogram=torch.load(utt.audio_spec_path),
            speaker_id=LongTensor([utt.speaker_id])
            if utt.speaker_id is not None
            else None,
            text=utt.text,
            word_spans=utt.word_spans,
            bert_features=bert_features,
        )

    @staticmethod
    def load_dataset(
        dataset_path: Path,
        max_phoneme_ids: Optional[int] = None,
        hop_length: int = 256,
    ) -> Iterable[Utterance]:
        if dataset_path.suffix.lower() == ".parquet":
            yield from PiperDataset.load_parquet_dataset(
                dataset_path,
                max_phoneme_ids=max_phoneme_ids,
                hop_length=hop_length,
            )
            return

        num_skipped = 0

        with open(dataset_path, "r", encoding="utf-8") as dataset_file:
            for line_idx, line in enumerate(dataset_file):
                line = line.strip()
                if not line:
                    continue

                try:
                    utt = PiperDataset.load_utterance(line, hop_length=hop_length)
                    if (max_phoneme_ids is None) or (
                        len(utt.phoneme_ids) <= max_phoneme_ids
                    ):
                        yield utt
                    else:
                        num_skipped += 1
                except Exception:
                    _LOGGER.exception(
                        "Error on line %s of %s: %s",
                        line_idx + 1,
                        dataset_path,
                        line,
                    )

        if num_skipped > 0:
            _LOGGER.warning("Skipped %s utterance(s)", num_skipped)

    @staticmethod
    def load_parquet_dataset(
        dataset_path: Path,
        max_phoneme_ids: Optional[int] = None,
        hop_length: int = 256,
    ) -> Iterable[Utterance]:
        import pyarrow.parquet as pq

        num_skipped = 0
        table = pq.read_table(dataset_path)
        for row_idx, utt_dict in enumerate(table.to_pylist()):
            try:
                utt = PiperDataset.load_utterance_dict(utt_dict, hop_length=hop_length)
                if (max_phoneme_ids is None) or (
                    len(utt.phoneme_ids) <= max_phoneme_ids
                ):
                    yield utt
                else:
                    num_skipped += 1
            except Exception:
                _LOGGER.exception(
                    "Error on row %s of %s: %s",
                    row_idx + 1,
                    dataset_path,
                    utt_dict,
                )

        if num_skipped > 0:
            _LOGGER.warning("Skipped %s utterance(s)", num_skipped)

    @staticmethod
    def load_utterance(line: str, hop_length: int = 256) -> Utterance:
        return PiperDataset.load_utterance_dict(json.loads(line), hop_length=hop_length)

    @staticmethod
    def load_utterance_dict(utt_dict: dict, hop_length: int = 256) -> Utterance:
        audio_path = utt_dict.get("audio_path")
        audio_norm_path = Path(utt_dict["audio_norm_path"])
        audio_spec_path = Path(utt_dict["audio_spec_path"])
        bert_path = (
            utt_dict.get("bert_path")
            or utt_dict.get("bert_feature_path")
            or utt_dict.get("semantic_features_path")
        )
        return Utterance(
            phoneme_ids=utt_dict["phoneme_ids"],
            audio_norm_path=audio_norm_path,
            audio_spec_path=audio_spec_path,
            audio_path=Path(audio_path) if audio_path else None,
            speaker_id=utt_dict.get("speaker_id"),
            text=utt_dict.get("text"),
            word_spans=utt_dict.get("word_spans"),
            bert_path=Path(bert_path) if bert_path else None,
            bert_dim=utt_dict.get("bert_dim"),
            spec_length=utt_dict.get("spec_length")
            or utt_dict.get("spectrogram_length")
            or _estimate_spec_length(
                Path(audio_path) if audio_path else None,
                audio_norm_path,
                audio_spec_path,
                hop_length=hop_length,
            ),
        )


def _estimate_spec_length(
    audio_path: Optional[Path],
    audio_norm_path: Path,
    audio_spec_path: Path,
    hop_length: int,
) -> int:
    if audio_path is not None and audio_path.exists():
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                return max(1, int(wav_file.getnframes()) // int(hop_length))
        except (wave.Error, OSError, EOFError):
            pass

    if audio_norm_path.exists():
        try:
            audio = torch.load(audio_norm_path, map_location="cpu")
            return max(1, int(audio.shape[-1]) // int(hop_length))
        except Exception:
            _LOGGER.debug("Failed to estimate length from %s", audio_norm_path, exc_info=True)

    if audio_spec_path.exists():
        try:
            spec = torch.load(audio_spec_path, map_location="cpu")
            return max(1, int(spec.shape[-1]))
        except Exception:
            _LOGGER.debug("Failed to estimate length from %s", audio_spec_path, exc_info=True)

    return 1


class LengthBucketBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Length bucket sampler for single-process training."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        boundaries: Sequence[int],
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.lengths = _dataset_lengths(dataset)
        self.batch_size = int(batch_size)
        self.boundaries = sorted(int(boundary) for boundary in boundaries)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.buckets = self._create_buckets()
        _LOGGER.info(
            "Length bucket sampler: batch_size=%s boundaries=%s bucket_sizes=%s",
            self.batch_size,
            self.boundaries,
            [len(bucket) for bucket in self.buckets],
        )

    def _create_buckets(self) -> List[List[int]]:
        buckets: List[List[int]] = [[] for _ in range(len(self.boundaries) - 1)]
        for index, length in enumerate(self.lengths):
            bucket_index = self._bucket_index(int(length))
            if bucket_index >= 0:
                buckets[bucket_index].append(index)
        buckets = [bucket for bucket in buckets if bucket]
        if not buckets:
            raise ValueError("Length bucket sampler has no non-empty buckets")
        return buckets

    def _bucket_index(self, length: int) -> int:
        for index in range(len(self.boundaries) - 1):
            if self.boundaries[index] < length <= self.boundaries[index + 1]:
                return index
        return -1

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        self.epoch += 1

        batches: List[List[int]] = []
        for bucket in self.buckets:
            if self.shuffle:
                order = torch.randperm(len(bucket), generator=generator).tolist()
            else:
                order = list(range(len(bucket)))
            indices = [bucket[index] for index in order]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)

        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for bucket in self.buckets:
            full, remainder = divmod(len(bucket), self.batch_size)
            total += full
            if remainder and not self.drop_last:
                total += 1
        return total


def _dataset_lengths(dataset: Dataset) -> List[int]:
    if hasattr(dataset, "lengths"):
        return [int(length) for length in getattr(dataset, "lengths")]

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent = getattr(dataset, "dataset")
        indices = getattr(dataset, "indices")
        if hasattr(parent, "lengths"):
            parent_lengths = getattr(parent, "lengths")
            return [int(parent_lengths[index]) for index in indices]

    raise ValueError("Dataset does not expose length metadata for bucketing")


class UtteranceCollate:
    def __init__(self, is_multispeaker: bool, segment_size: int):
        self.is_multispeaker = is_multispeaker
        self.segment_size = segment_size

    def __call__(self, utterances: Sequence[UtteranceTensors]) -> Batch:
        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = 0
        max_spec_length = 0
        max_audio_length = 0

        num_mels = 0

        # Determine lengths
        for utt_idx, utt in enumerate(utterances):
            assert utt.spectrogram is not None
            assert utt.audio_norm is not None

            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            max_phonemes_length = max(max_phonemes_length, phoneme_length)
            max_spec_length = max(max_spec_length, spec_length)
            max_audio_length = max(max_audio_length, audio_length)

            num_mels = utt.spectrogram.size(0)
            if self.is_multispeaker:
                assert utt.speaker_id is not None, "Missing speaker id"

        # Audio cannot be smaller than segment size (8192)
        max_audio_length = max(max_audio_length, self.segment_size)

        # Create padded tensors
        phonemes_padded = LongTensor(num_utterances, max_phonemes_length)
        spec_padded = FloatTensor(num_utterances, num_mels, max_spec_length)
        audio_padded = FloatTensor(num_utterances, 1, max_audio_length)

        phonemes_padded.zero_()
        spec_padded.zero_()
        audio_padded.zero_()

        phoneme_lengths = LongTensor(num_utterances)
        spec_lengths = LongTensor(num_utterances)
        audio_lengths = LongTensor(num_utterances)

        speaker_ids: Optional[LongTensor] = None
        if self.is_multispeaker:
            speaker_ids = LongTensor(num_utterances)

        texts: List[str] = []
        word_spans: List[Optional[List[List[int]]]] = []
        has_bert_features = any(utt.bert_features is not None for utt in utterances)
        if has_bert_features and any(utt.bert_features is None for utt in utterances):
            raise ValueError("Batch mixes utterances with and without precomputed BERT features")

        bert_padded: Optional[FloatTensor] = None
        bert_dim = 0
        if has_bert_features:
            first_features = next(utt.bert_features for utt in utterances if utt.bert_features is not None)
            assert first_features is not None
            bert_dim = int(first_features.size(0))
            bert_padded = FloatTensor(num_utterances, bert_dim, max_phonemes_length)
            bert_padded.zero_()

        # Sort by decreasing spectrogram length
        sorted_utterances = sorted(
            utterances, key=lambda u: u.spectrogram.size(1), reverse=True
        )
        for utt_idx, utt in enumerate(sorted_utterances):
            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            phonemes_padded[utt_idx, :phoneme_length] = utt.phoneme_ids
            phoneme_lengths[utt_idx] = phoneme_length

            spec_padded[utt_idx, :, :spec_length] = utt.spectrogram
            spec_lengths[utt_idx] = spec_length

            audio_padded[utt_idx, :, :audio_length] = utt.audio_norm
            audio_lengths[utt_idx] = audio_length

            if utt.speaker_id is not None:
                assert speaker_ids is not None
                speaker_ids[utt_idx] = utt.speaker_id
            # Preserve original text (if present) for optional semantic encoders
            texts.append(utt.text or "")
            word_spans.append(utt.word_spans)

            if bert_padded is not None:
                assert utt.bert_features is not None
                features = utt.bert_features
                if features.size(0) != bert_dim:
                    raise ValueError(
                        f"BERT feature dim mismatch: expected {bert_dim}, got {features.size(0)}"
                    )
                copy_len = min(int(features.size(1)), int(phoneme_length))
                bert_padded[utt_idx, :, :copy_len] = features[:, :copy_len]

        if _DEBUG_SEMANTIC and texts:
            first_text = texts[0]
            first_utt = sorted_utterances[0]
            _LOGGER.debug(
                "UtteranceCollate: batch_size=%s, example_text[0]=%r, "
                "phoneme_len[0]=%s, spec_len[0]=%s, audio_len[0]=%s",
                num_utterances,
                first_text,
                first_utt.phoneme_ids.size(0),
                first_utt.spectrogram.size(1),
                first_utt.audio_norm.size(1),
            )

        return Batch(
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            audios=audio_padded,
            audio_lengths=audio_lengths,
            speaker_ids=speaker_ids,
            texts=texts,
            word_spans=word_spans,
            bert_features=bert_padded,
        )
