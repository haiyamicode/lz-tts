"""Production Starling inference helpers.

This module keeps the API server thin while still reusing the local Matcha-TTS
model package/checkpoints during the transition to production packaging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

_LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHA_ROOT = PROJECT_ROOT / "local" / "Matcha-TTS"


class MatchaSettings(Protocol):
    device: str
    checkpoint: str
    config_path: str
    semantic_model_name: str
    semantic_max_tokens: int | None
    icbpe_vocab_path: str
    phoneme_vocab_path: str
    filelist_path: str
    max_batch_size: int
    batch_wait_ms: float
    n_timesteps: int
    temperature: float
    length_scale: float
    sample_rate: int
    vocoder: str
    n_mels: int
    n_fft: int
    hop_length: int
    win_length: int
    f_min: float
    f_max: float | None
    mel_mean: float
    mel_std: float


class MatchaRequestLike(Protocol):
    text: str
    language: str
    input_type: str
    speaker_id: int | None
    neural: bool
    steps: int | None
    temperature: float | None
    length_scale: float | None


@dataclass
class MatchaPreparedItem:
    text: str
    language: str
    phoneme_text: str
    x_phoneme: Any
    semantic_features: Any
    speaker_id: int


@dataclass
class MatchaBatchRequest:
    text: str
    language: str
    input_type: str
    speaker_id: int | None
    neural: bool
    steps: int | None
    temperature: float | None
    length_scale: float | None
    future: asyncio.Future
    queued_at: float


@dataclass
class MatchaBatchResult:
    audio: np.ndarray
    sample_rate: int
    audio_seconds: float
    backend_seconds: float
    model_rtf: float
    backend_rtf: float
    batch_size: int
    queue_seconds: float
    text: str
    phoneme_text: str


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return MATCHA_ROOT / path


class MatchaBackend:
    """Batched Starling inference backend."""

    def __init__(self, settings: MatchaSettings):
        self.settings = settings
        self.sample_rate = int(settings.sample_rate)
        self.device_name = settings.device
        self.checkpoint_path = _resolve_path(settings.checkpoint)
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Starling checkpoint not found: {self.checkpoint_path}")
        if str(MATCHA_ROOT) not in sys.path:
            sys.path.insert(0, str(MATCHA_ROOT))

        import torch  # pylint: disable=import-outside-toplevel
        from matcha.models.matcha_tts import MatchaTTS  # pylint: disable=import-outside-toplevel
        from transformers import AutoModel  # pylint: disable=import-outside-toplevel
        from src.piper.hf_cache import resolve_hf_model_path  # pylint: disable=import-outside-toplevel
        from src.piper.semantic import SemanticTokenizer  # pylint: disable=import-outside-toplevel

        self.torch = torch
        self.device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        self.model = MatchaTTS.load_from_checkpoint(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=False,
        ).to(self.device).eval()
        # The service config is the source of truth for vocoder/mel denormalization.
        self.model.update_data_statistics(
            {"mel_mean": float(self.settings.mel_mean), "mel_std": float(self.settings.mel_std)}
        )

        self.vocoder_kind = self.settings.vocoder
        self.vocoder = self._load_vocoder()

        self.config_path = _resolve_path(self.settings.config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Starling preprocessing config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.voice_config = json.load(handle)
        self.speaker_id_map = {
            str(label): int(speaker_id)
            for label, speaker_id in (self.voice_config.get("speaker_id_map") or {}).items()
        }

        semantic_model_name = self.settings.semantic_model_name or "distilbert/distilbert-base-multilingual-cased"
        semantic_path = resolve_hf_model_path(semantic_model_name, require_weights=True)
        self.semantic_tokenizer = SemanticTokenizer(semantic_model_name, max_length=self.settings.semantic_max_tokens)
        self.semantic_model = AutoModel.from_pretrained(semantic_path).to(self.device).eval()

        expected_vocab_size = int(self.model.encoder.emb.weight.shape[0])
        if expected_vocab_size != int(self.model.n_vocab):
            raise ValueError(f"Starling checkpoint vocab mismatch: encoder={expected_vocab_size} model={self.model.n_vocab}")

        _LOGGER.info(
            "Loaded Starling backend: checkpoint=%s device=%s vocoder=%s sr=%d n_mels=%d n_vocab=%d speakers=%d semantic=%s",
            self.checkpoint_path,
            self.device,
            self.vocoder_kind,
            self.sample_rate,
            int(self.settings.n_mels),
            int(self.model.n_vocab),
            len(self.speaker_id_map),
            semantic_model_name,
        )

    def _load_vocoder(self):
        if self.vocoder_kind != "vocos24k":
            raise ValueError(f"Unsupported Starling vocoder: {self.vocoder_kind}")
        from vocos import Vocos  # pylint: disable=import-outside-toplevel

        return Vocos.from_pretrained("charactr/vocos-mel-24khz").eval().to(self.device)

    def _speaker_label_for_language(self, language: str) -> str:
        base = language.lower().split("-")[0] or "en"
        if base in self.speaker_id_map:
            return base
        if language in self.speaker_id_map:
            return language
        return "en" if "en" in self.speaker_id_map else next(iter(self.speaker_id_map), "en")

    def _phonemize_request(self, request: MatchaBatchRequest) -> dict[str, Any]:
        from src.piper.preprocess import phonemize_text_for_speaker  # pylint: disable=import-outside-toplevel

        language = request.language.lower().split("-")[0] or "en"
        speaker_label = self._speaker_label_for_language(language)
        return phonemize_text_for_speaker(
            request.text,
            self.config_path,
            speaker_label=speaker_label,
            neural=request.neural,
        )

    def _prepare_batch(self, requests: list[MatchaBatchRequest]) -> list[MatchaPreparedItem]:
        from src.piper.semantic import align_phone_features, build_bert_input  # pylint: disable=import-outside-toplevel

        rows = [self._phonemize_request(request) for request in requests]
        texts = [str(row["text"]) for row in rows]
        lengths = [len(row["phoneme_ids"]) for row in rows]
        word_spans = [row.get("word_spans") for row in rows]
        bert_input = build_bert_input(
            texts,
            self.semantic_tokenizer,
            phoneme_lengths=lengths,
            word_spans=word_spans,
        )
        if bert_input is None or "word2ph" not in bert_input:
            raise ValueError("Failed to build Starling semantic input")
        input_ids = bert_input["input_ids"].to(self.device)
        attention_mask = bert_input["attention_mask"].to(self.device)
        word2ph = bert_input["word2ph"].to(self.device)

        with self.torch.inference_mode():
            hidden = self.semantic_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        prepared: list[MatchaPreparedItem] = []
        for request, row, item_hidden, item_word2ph, phone_len in zip(requests, rows, hidden, word2ph, lengths):
            language = request.language.lower().split("-")[0] or "en"
            phoneme_ids = row["phoneme_ids"]
            phoneme_text = "".join(row.get("phonemes") or [])
            speaker_id = request.speaker_id if request.speaker_id is not None else int(row.get("speaker_id", 0))
            semantic_features = align_phone_features(item_hidden, item_word2ph, phone_len)
            prepared.append(
                MatchaPreparedItem(
                    text=str(row["text"]),
                    language=language,
                    phoneme_text=phoneme_text,
                    x_phoneme=self.torch.tensor(phoneme_ids, dtype=self.torch.long),
                    semantic_features=semantic_features.detach().cpu(),
                    speaker_id=speaker_id,
                )
            )
        return prepared

    def _prepare_item(self, request: MatchaBatchRequest) -> MatchaPreparedItem:
        return self._prepare_batch([request])[0]

    def prepare_row(self, row: list[str]) -> MatchaPreparedItem:
        _filepath, _lang, _phoneme_text, _raw_text = row[:4]
        raise NotImplementedError("Starling production inference no longer supports aligned filelist rows")

    def synthesize_prepared_batch(
        self,
        prepared: list[MatchaPreparedItem],
        *,
        steps: int | None = None,
        temperature: float | None = None,
        length_scale: float | None = None,
        input_type: Literal["aligned"] = "aligned",
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        torch = self.torch
        started = time.perf_counter()
        batch_size = len(prepared)
        max_len = max(item.x_phoneme.shape[-1] for item in prepared)

        x = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        semantic_features = torch.zeros(
            (batch_size, int(prepared[0].semantic_features.shape[0]), max_len),
            dtype=torch.float32,
            device=self.device,
        )
        x_lengths = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        spks = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        for idx, item in enumerate(prepared):
            length = item.x_phoneme.shape[-1]
            x_lengths[idx] = length
            spks[idx] = item.speaker_id
            x[idx, :length] = item.x_phoneme.to(self.device)
            semantic_features[idx, :, :length] = item.semantic_features.to(self.device)

        steps = steps or self.settings.n_timesteps
        temperature = temperature or self.settings.temperature
        length_scale = length_scale or self.settings.length_scale

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with torch.inference_mode():
            output = self.model.synthesise(
                x,
                x_lengths,
                n_timesteps=steps,
                temperature=temperature,
                length_scale=length_scale,
                spks=spks,
                semantic_features=semantic_features,
            )
            audio = self.vocoder.decode(output["mel"]).clamp(-1, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed = time.perf_counter() - started
        audio = audio.detach().float().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        wavs = []
        for idx, mel_length in enumerate(output["mel_lengths"].detach().cpu().tolist()):
            audio_samples = max(1, int(mel_length) * int(self.settings.hop_length))
            current = audio[idx, : min(audio_samples, audio.shape[-1])].numpy()
            wavs.append(np.clip(current, -1.0, 1.0))

        return wavs, {"elapsed": elapsed, "output": output}

    def synthesize_batch(self, requests: list[MatchaBatchRequest]) -> list[MatchaBatchResult]:
        started = time.perf_counter()
        prepared = self._prepare_batch(requests)
        wavs, info = self.synthesize_prepared_batch(
            prepared,
            steps=requests[0].steps,
            temperature=requests[0].temperature,
            length_scale=requests[0].length_scale,
            input_type=requests[0].input_type,
        )
        elapsed = info["elapsed"]
        output = info["output"]

        total_audio_seconds = 0.0
        trimmed_audio = []
        for current in wavs:
            audio_int16 = (current * 32767.0).astype(np.int16)
            audio_seconds = float(audio_int16.shape[-1]) / self.sample_rate
            total_audio_seconds += audio_seconds
            trimmed_audio.append((audio_int16, audio_seconds))

        backend_rtf = elapsed / total_audio_seconds if total_audio_seconds else 0.0
        results: list[MatchaBatchResult] = []
        for request, item, (audio_int16, audio_seconds) in zip(requests, prepared, trimmed_audio):
            results.append(
                MatchaBatchResult(
                    audio=audio_int16,
                    sample_rate=self.sample_rate,
                    audio_seconds=audio_seconds,
                    backend_seconds=elapsed,
                    model_rtf=float(output["rtf"]),
                    backend_rtf=backend_rtf,
                    batch_size=len(prepared),
                    queue_seconds=max(0.0, started - request.queued_at),
                    text=item.text,
                    phoneme_text=item.phoneme_text,
                )
            )
        return results


class MatchaBatcher:
    def __init__(self, backend: MatchaBackend, settings: MatchaSettings):
        self.backend = backend
        self.settings = settings
        self._queue: list[MatchaBatchRequest] = []
        self._condition = asyncio.Condition()
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def submit(self, request: MatchaRequestLike) -> MatchaBatchResult:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = MatchaBatchRequest(
            text=request.text,
            language=request.language,
            input_type=request.input_type,
            speaker_id=request.speaker_id,
            neural=request.neural,
            steps=request.steps,
            temperature=request.temperature,
            length_scale=request.length_scale,
            future=future,
            queued_at=time.perf_counter(),
        )
        async with self._condition:
            self._queue.append(item)
            self._condition.notify()
        return await future

    async def _worker(self) -> None:
        while True:
            async with self._condition:
                while not self._queue:
                    await self._condition.wait()
            wait_seconds = self.settings.batch_wait_ms / 1000.0
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            async with self._condition:
                batch = self._queue[: self.settings.max_batch_size]
                del self._queue[: len(batch)]

            try:
                results = await asyncio.to_thread(self.backend.synthesize_batch, batch)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _LOGGER.exception("Starling batch failed")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
                continue

            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)
