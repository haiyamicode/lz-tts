import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from src.starling.utils.audio import mel_spectrogram, normalize_audio_rms, vocos_mel_spectrogram
from src.starling.utils.model import fix_len_compatibility, normalize
from src.starling.utils.utils import intersperse


def sample_same_utterance_reference(
    audio,
    sample_rate,
    min_ratio,
    max_ratio,
    short_threshold_seconds,
    short_ratio,
    randomize=True,
    rng=None,
):
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if not 0 < min_ratio <= max_ratio <= 1:
        raise ValueError(
            f"Reference ratios must satisfy 0 < min <= max <= 1, got {min_ratio} and {max_ratio}"
        )
    if short_threshold_seconds < 0:
        raise ValueError("short_threshold_seconds must be non-negative")
    if not 0 < short_ratio <= 1:
        raise ValueError(f"short_ratio must be in (0, 1], got {short_ratio}")

    total_samples = int(audio.shape[-1])
    if total_samples <= 0:
        raise ValueError("Cannot sample a reference from empty audio")
    rng = rng or random
    duration_seconds = total_samples / sample_rate
    if duration_seconds <= short_threshold_seconds:
        ratio = short_ratio
    elif randomize:
        ratio = rng.uniform(min_ratio, max_ratio)
    else:
        ratio = (min_ratio + max_ratio) / 2

    reference_samples = min(total_samples, max(1, round(total_samples * ratio)))
    max_start = total_samples - reference_samples
    if randomize and max_start > 0:
        start = rng.randint(0, max_start)
    else:
        start = max_start // 2
    return audio[..., start : start + reference_samples]


def parse_filelist(filelist_path, split_char="|"):
    filelist_path = Path(filelist_path)
    with open(filelist_path, encoding="utf-8") as f:
        if filelist_path.suffix == ".jsonl":
            filepaths_and_text = [json.loads(line) for line in f if line.strip()]
        else:
            filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
        mel_backend="matcha",
        prompt_mel_enabled=False,
        prompt_mel_same_speaker_prob=1.0,
        prompt_mel_same_utterance_prob=0.05,
        prompt_embedding_enabled=False,
        prompt_embedding_dim=192,
        use_same_utterance_as_reference=False,
        same_utterance_reference_min_ratio=0.2,
        same_utterance_reference_max_ratio=0.5,
        same_utterance_reference_short_threshold_seconds=5.0,
        same_utterance_reference_short_ratio=0.5,
        same_utterance_reference_embedding_count=1,
        speaker_auto_id=0,
        speaker_auto_train_prob=0.0,
        rms_normalize_audio=False,
        rms_target=0.1,
        rms_peak_limit=0.99,
        rms_eps=1e-6,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            self.hparams.mel_backend,
            self.hparams.prompt_mel_enabled,
            self.hparams.prompt_mel_same_speaker_prob,
            self.hparams.prompt_mel_same_utterance_prob,
            self.hparams.prompt_embedding_enabled,
            self.hparams.prompt_embedding_dim,
            self.hparams.use_same_utterance_as_reference,
            self.hparams.same_utterance_reference_min_ratio,
            self.hparams.same_utterance_reference_max_ratio,
            self.hparams.same_utterance_reference_short_threshold_seconds,
            self.hparams.same_utterance_reference_short_ratio,
            self.hparams.same_utterance_reference_embedding_count,
            True,
            self.hparams.speaker_auto_id,
            self.hparams.speaker_auto_train_prob,
            self.hparams.rms_normalize_audio,
            self.hparams.rms_target,
            self.hparams.rms_peak_limit,
            self.hparams.rms_eps,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            self.hparams.mel_backend,
            self.hparams.prompt_mel_enabled,
            1.0,
            0.0,
            self.hparams.prompt_embedding_enabled,
            self.hparams.prompt_embedding_dim,
            self.hparams.use_same_utterance_as_reference,
            self.hparams.same_utterance_reference_min_ratio,
            self.hparams.same_utterance_reference_max_ratio,
            self.hparams.same_utterance_reference_short_threshold_seconds,
            self.hparams.same_utterance_reference_short_ratio,
            self.hparams.same_utterance_reference_embedding_count,
            False,
            self.hparams.speaker_auto_id,
            0.0,
            self.hparams.rms_normalize_audio,
            self.hparams.rms_target,
            self.hparams.rms_peak_limit,
            self.hparams.rms_eps,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            persistent_workers=bool(
                self.hparams.num_workers and self.hparams.use_same_utterance_as_reference
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            persistent_workers=bool(
                self.hparams.num_workers and self.hparams.use_same_utterance_as_reference
            ),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
        mel_backend="matcha",
        prompt_mel_enabled=False,
        prompt_mel_same_speaker_prob=1.0,
        prompt_mel_same_utterance_prob=0.05,
        prompt_embedding_enabled=False,
        prompt_embedding_dim=192,
        use_same_utterance_as_reference=False,
        same_utterance_reference_min_ratio=0.2,
        same_utterance_reference_max_ratio=0.5,
        same_utterance_reference_short_threshold_seconds=5.0,
        same_utterance_reference_short_ratio=0.5,
        same_utterance_reference_embedding_count=1,
        randomize_same_utterance_reference=True,
        speaker_auto_id=0,
        speaker_auto_prob=0.0,
        rms_normalize_audio=False,
        rms_target=0.1,
        rms_peak_limit=0.99,
        rms_eps=1e-6,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.mel_backend = mel_backend
        self.prompt_mel_enabled = prompt_mel_enabled
        self.prompt_mel_same_speaker_prob = float(prompt_mel_same_speaker_prob)
        self.prompt_mel_same_utterance_prob = float(prompt_mel_same_utterance_prob)
        self.prompt_embedding_enabled = prompt_embedding_enabled
        self.prompt_embedding_dim = int(prompt_embedding_dim)
        self.use_same_utterance_as_reference = bool(use_same_utterance_as_reference)
        self.same_utterance_reference_min_ratio = float(same_utterance_reference_min_ratio)
        self.same_utterance_reference_max_ratio = float(same_utterance_reference_max_ratio)
        self.same_utterance_reference_short_threshold_seconds = float(
            same_utterance_reference_short_threshold_seconds
        )
        self.same_utterance_reference_short_ratio = float(same_utterance_reference_short_ratio)
        self.same_utterance_reference_embedding_count = int(
            same_utterance_reference_embedding_count
        )
        self.randomize_same_utterance_reference = bool(randomize_same_utterance_reference)
        self.speaker_auto_id = int(speaker_auto_id)
        self.speaker_auto_prob = float(speaker_auto_prob)
        self.rms_normalize_audio = bool(rms_normalize_audio)
        self.rms_target = float(rms_target)
        self.rms_peak_limit = float(rms_peak_limit)
        self.rms_eps = float(rms_eps)
        if not 0.0 <= self.speaker_auto_prob <= 1.0:
            raise ValueError(f"speaker_auto_prob must be between 0 and 1, got {self.speaker_auto_prob}")
        if self.n_spks > 1 and not 0 <= self.speaker_auto_id < self.n_spks:
            raise ValueError(f"speaker_auto_id={self.speaker_auto_id} is outside n_spks={self.n_spks}")
        if not (
            0
            < self.same_utterance_reference_min_ratio
            <= self.same_utterance_reference_max_ratio
            <= 1
        ):
            raise ValueError(
                "same-utterance reference ratios must satisfy "
                f"0 < min <= max <= 1, got {self.same_utterance_reference_min_ratio} and "
                f"{self.same_utterance_reference_max_ratio}"
            )
        if not 0 < self.same_utterance_reference_short_ratio <= 1:
            raise ValueError(
                "same_utterance_reference_short_ratio must be in (0, 1], got "
                f"{self.same_utterance_reference_short_ratio}"
            )
        if self.same_utterance_reference_short_threshold_seconds < 0:
            raise ValueError("same_utterance_reference_short_threshold_seconds must be non-negative")
        if self.same_utterance_reference_embedding_count < 1:
            raise ValueError("same_utterance_reference_embedding_count must be at least 1")
        if self.use_same_utterance_as_reference and not (
            self.prompt_mel_enabled or self.prompt_embedding_enabled
        ):
            raise ValueError(
                "use_same_utterance_as_reference requires prompt mel or prompt embedding conditioning"
            )
        self._prompt_rng = random.Random(seed)
        self._speaker_rng = random.Random(None if seed is None else int(seed) + 1)
        self.prompt_candidates = []
        self.prompt_candidates_by_speaker = {}

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        if self.prompt_mel_enabled and not self.use_same_utterance_as_reference:
            self._build_prompt_candidates()
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def _speaker_from_row(self, filepath_and_text):
        if isinstance(filepath_and_text, dict):
            speaker = filepath_and_text.get("speaker")
            if speaker is not None:
                return str(speaker)
            speaker_id = filepath_and_text.get("speaker_id")
            return str(speaker_id) if speaker_id is not None else None
        if self.n_spks > 1 and len(filepath_and_text) > 1:
            return str(filepath_and_text[1])
        return None

    def _audio_path_from_row(self, filepath_and_text):
        if isinstance(filepath_and_text, dict):
            return filepath_and_text["audio_path"]
        return filepath_and_text[0]

    def _maybe_use_auto_speaker(self, spk):
        if spk is None:
            return spk
        if self.n_spks > 1 and not 0 <= int(spk) < self.n_spks:
            raise ValueError(f"speaker_id={spk} is outside n_spks={self.n_spks}")
        if self.speaker_auto_prob <= 0:
            return spk
        if self._speaker_rng.random() < self.speaker_auto_prob:
            return self.speaker_auto_id
        return spk

    def _build_prompt_candidates(self):
        candidates = {}
        all_candidates = []
        for row in self.filepaths_and_text:
            audio_path = self._audio_path_from_row(row)
            all_candidates.append(audio_path)
            speaker = self._speaker_from_row(row)
            if speaker is None:
                continue
            candidates.setdefault(speaker, []).append(audio_path)
        self.prompt_candidates = all_candidates
        self.prompt_candidates_by_speaker = candidates

    def get_prompt_mel(self, filepath_and_text, filepath):
        if not self.prompt_mel_enabled:
            return None
        if isinstance(filepath_and_text, dict):
            explicit_prompt = filepath_and_text.get("prompt_mel_path") or filepath_and_text.get(
                "prompt_audio_path"
            )
            if explicit_prompt:
                return self.get_mel(explicit_prompt)
        speaker = self._speaker_from_row(filepath_and_text)
        prompt_path = filepath
        if speaker is not None and self._prompt_rng.random() < self.prompt_mel_same_speaker_prob:
            candidates = self.prompt_candidates_by_speaker.get(speaker) or []
        else:
            candidates = self.prompt_candidates
        if candidates and self._prompt_rng.random() >= self.prompt_mel_same_utterance_prob:
            alternatives = [candidate for candidate in candidates if candidate != filepath]
            prompt_path = self._prompt_rng.choice(alternatives or candidates)
        return self.get_mel(prompt_path)

    def get_prompt_embedding(self, filepath_and_text, randomize_bank=True):
        if not self.prompt_embedding_enabled:
            return None
        embedding_path = None
        if isinstance(filepath_and_text, dict):
            embedding_path = (
                filepath_and_text.get("prompt_embedding_path")
                or filepath_and_text.get("speaker_embedding_path")
                or filepath_and_text.get("embedding_path")
            )
        elif len(filepath_and_text) >= 6:
            embedding_path = filepath_and_text[5]
        if not embedding_path:
            raise ValueError(
                "prompt_embedding_enabled requires prompt_embedding_path, "
                "speaker_embedding_path, or a 6th filelist field"
            )
        embedding_path = Path(embedding_path)
        if embedding_path.suffix == ".pt":
            embedding = torch.load(embedding_path, map_location="cpu", weights_only=True)
            embedding = embedding.float()
        else:
            embedding = torch.from_numpy(np.load(embedding_path).astype(np.float32))
        if embedding.dim() == 2:
            if (
                self.use_same_utterance_as_reference
                and embedding.shape[0] != self.same_utterance_reference_embedding_count
            ):
                raise ValueError(
                    f"Expected {self.same_utterance_reference_embedding_count} prompt embeddings "
                    f"at {embedding_path}, got {embedding.shape[0]}"
                )
            embedding_index = (
                self._prompt_rng.randrange(embedding.shape[0]) if randomize_bank else 0
            )
            embedding = embedding[embedding_index]
        if embedding.dim() != 1:
            raise ValueError(f"Expected 1D prompt embedding at {embedding_path}, got {tuple(embedding.shape)}")
        if embedding.shape[0] != self.prompt_embedding_dim:
            raise ValueError(f"Expected prompt embedding dim {self.prompt_embedding_dim}, got {embedding.shape[0]}")
        return embedding

    def _load_audio(self, filepath):
        filepath = Path(filepath)
        if filepath.suffix == ".pt":
            audio = torch.load(filepath, map_location="cpu", weights_only=True).float()
            sr = self.sample_rate
        else:
            audio, sr = ta.load(filepath)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            raise ValueError(f"Expected sample_rate={self.sample_rate}, got {sr} for {filepath}")
        if self.rms_normalize_audio:
            audio = normalize_audio_rms(audio, self.rms_target, self.rms_peak_limit, self.rms_eps)
        return audio.contiguous()

    def _same_utterance_reference_audio(self, audio):
        return sample_same_utterance_reference(
            audio,
            self.sample_rate,
            self.same_utterance_reference_min_ratio,
            self.same_utterance_reference_max_ratio,
            self.same_utterance_reference_short_threshold_seconds,
            self.same_utterance_reference_short_ratio,
            randomize=self.randomize_same_utterance_reference,
            rng=self._prompt_rng,
        )

    def get_same_utterance_reference(self, filepath_and_text, audio):
        audio = self._same_utterance_reference_audio(audio)
        prompt_mel = self._audio_to_mel(audio) if self.prompt_mel_enabled else None
        prompt_embedding = self.get_prompt_embedding(
            filepath_and_text,
            randomize_bank=self.randomize_same_utterance_reference,
        )
        return prompt_mel, prompt_embedding

    def get_datapoint(self, filepath_and_text):
        bert_features = None
        if isinstance(filepath_and_text, dict):
            filepath = filepath_and_text["audio_path"]
            text = torch.LongTensor(filepath_and_text["phoneme_ids"])
            cleaned_text = filepath_and_text.get("text", "")
            spk = filepath_and_text.get("speaker_id")
            if spk is not None:
                spk = int(spk)
            bert_path = filepath_and_text.get("bert_path")
            if bert_path:
                bert_features = torch.load(bert_path, map_location="cpu", weights_only=True).float()
                if bert_features.dim() != 2:
                    raise ValueError(f"Expected 2D BERT features at {bert_path}, got {tuple(bert_features.shape)}")
                if bert_features.shape[-1] != text.shape[-1]:
                    raise ValueError(
                        f"BERT/text length mismatch for {filepath}: "
                        f"bert_len={bert_features.shape[-1]} text_len={text.shape[-1]}"
                    )
        elif self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
            text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None
            text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        spk = self._maybe_use_auto_speaker(spk)

        durations = self.get_durations(filepath, text) if self.load_durations else None

        if self.use_same_utterance_as_reference:
            audio = self._load_audio(filepath)
            mel = self._audio_to_mel(audio)
            prompt_mel, prompt_embedding = self.get_same_utterance_reference(
                filepath_and_text, audio
            )
        else:
            mel = self.get_mel(filepath)
            prompt_mel = self.get_prompt_mel(filepath_and_text, filepath)
            prompt_embedding = self.get_prompt_embedding(filepath_and_text)

        return {
            "x": text,
            "y": mel,
            "spk": spk,
            "filepath": filepath,
            "x_text": cleaned_text,
            "durations": durations,
            "bert_features": bert_features,
            "prompt_mel": prompt_mel,
            "prompt_embedding": prompt_embedding,
        }

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}; "
                "generate durations before enabling load_durations."
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, filepath):
        return self._audio_to_mel(self._load_audio(filepath))

    def _audio_to_mel(self, audio):
        if self.mel_backend == "vocos_mel_24khz":
            mel = vocos_mel_spectrogram(
                audio,
                sampling_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_size=self.hop_length,
                num_mels=self.n_mels,
            ).squeeze()
        else:
            mel = mel_spectrogram(
                audio,
                self.n_fft,
                self.n_mels,
                self.sample_rate,
                self.hop_length,
                self.win_length,
                self.f_min,
                self.f_max,
                center=False,
            ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_text(self, text, add_blank=True):
        from src.starling.text import text_to_sequence

        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]
        has_bert_features = any(item.get("bert_features") is not None for item in batch)
        if has_bert_features and any(item.get("bert_features") is None for item in batch):
            raise ValueError("Batch mixes examples with and without BERT features")
        has_prompt_mel = any(item.get("prompt_mel") is not None for item in batch)
        if has_prompt_mel and any(item.get("prompt_mel") is None for item in batch):
            raise ValueError("Batch mixes examples with and without prompt mel")
        has_prompt_embedding = any(item.get("prompt_embedding") is not None for item in batch)
        if has_prompt_embedding and any(item.get("prompt_embedding") is None for item in batch):
            raise ValueError("Batch mixes examples with and without prompt embeddings")

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)
        bert_features = None
        if has_bert_features:
            bert_dim = batch[0]["bert_features"].shape[0]
            bert_features = torch.zeros((B, bert_dim, x_max_length), dtype=torch.float32)
        prompt_mel = None
        prompt_mel_lengths = None
        if has_prompt_mel:
            prompt_mel_max_length = fix_len_compatibility(max(item["prompt_mel"].shape[-1] for item in batch))
            prompt_mel = torch.zeros((B, n_feats, prompt_mel_max_length), dtype=torch.float32)
            prompt_mel_lengths = torch.zeros((B,), dtype=torch.long)
        prompt_embedding = None
        if has_prompt_embedding:
            prompt_embedding_dim = batch[0]["prompt_embedding"].shape[0]
            prompt_embedding = torch.zeros((B, prompt_embedding_dim), dtype=torch.float32)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]
            if bert_features is not None:
                features = item["bert_features"]
                if features.shape[0] != bert_features.shape[1]:
                    raise ValueError(
                        f"BERT feature dim mismatch: {features.shape[0]} != {bert_features.shape[1]}"
                    )
                bert_features[i, :, : features.shape[-1]] = features
            if prompt_mel is not None:
                prompt = item["prompt_mel"]
                if prompt.shape[0] != prompt_mel.shape[1]:
                    raise ValueError(f"Prompt mel dim mismatch: {prompt.shape[0]} != {prompt_mel.shape[1]}")
                prompt_mel[i, :, : prompt.shape[-1]] = prompt
                prompt_mel_lengths[i] = prompt.shape[-1]
            if prompt_embedding is not None:
                embedding = item["prompt_embedding"]
                if embedding.shape[0] != prompt_embedding.shape[1]:
                    raise ValueError(
                        f"Prompt embedding dim mismatch: {embedding.shape[0]} != {prompt_embedding.shape[1]}"
                    )
                prompt_embedding[i] = embedding

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
            "bert_features": bert_features,
            "prompt_mel": prompt_mel,
            "prompt_mel_lengths": prompt_mel_lengths,
            "prompt_embedding": prompt_embedding,
        }
