"""Production Matcha-TTS inference helpers.

This module keeps the API server thin while still reusing the local Matcha-TTS
model package/checkpoints during the transition to production packaging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import unicodedata
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
    x_text_tokens: Any
    x_phoneme_mask: Any
    x_text_mask: Any
    x_unit_ids: Any
    x_unit_texts: list[str]
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
    if path.parts and path.parts[0] == "local":
        return PROJECT_ROOT / path
    return MATCHA_ROOT / path


def _vocab_size(vocab_path: Path) -> int:
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "size" in payload:
        return int(payload["size"])
    if isinstance(payload, dict):
        return len(payload)
    raise ValueError(f"Unsupported vocab format: {vocab_path}")


class MatchaBackend:
    """Batched Matcha-TTS inference backend."""

    def __init__(self, settings: MatchaSettings):
        self.settings = settings
        self.sample_rate = int(settings.sample_rate)
        self.device_name = settings.device
        self.checkpoint_path = _resolve_path(settings.checkpoint)
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Matcha checkpoint not found: {self.checkpoint_path}")
        if str(MATCHA_ROOT) not in sys.path:
            sys.path.insert(0, str(MATCHA_ROOT))

        import torch  # pylint: disable=import-outside-toplevel
        from matcha.data.text_mel_datamodule import TextMelDataset  # pylint: disable=import-outside-toplevel
        from matcha.models.matcha_tts import MatchaTTSFusedSemantic  # pylint: disable=import-outside-toplevel

        self.torch = torch
        self.device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        self.model = MatchaTTSFusedSemantic.load_from_checkpoint(
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

        self.dataset = TextMelDataset(
            str(_resolve_path(self.settings.filelist_path)),
            n_spks=1,
            cleaners=["english_cleaners2"],
            add_blank=True,
            n_fft=int(self.settings.n_fft),
            n_mels=int(self.settings.n_mels),
            sample_rate=self.sample_rate,
            hop_length=int(self.settings.hop_length),
            win_length=int(self.settings.win_length),
            f_min=float(self.settings.f_min),
            f_max=self.settings.f_max,
            data_parameters={"mel_mean": float(self.settings.mel_mean), "mel_std": float(self.settings.mel_std)},
            seed=1234,
            icbpe_vocab_path=str(_resolve_path(self.settings.icbpe_vocab_path)),
            phoneme_vocab_path=str(_resolve_path(self.settings.phoneme_vocab_path)),
            language_id_map={},
            language_auto_id=0,
            language_auto_prob=0.0,
            mel_backend="vocos_mel_24khz",
        )

        expected_phoneme_vocab_size = int(self.model.encoder.phoneme_emb.weight.shape[0])
        actual_phoneme_vocab_size = _vocab_size(_resolve_path(self.settings.phoneme_vocab_path))
        if actual_phoneme_vocab_size != expected_phoneme_vocab_size:
            raise ValueError(
                "Phoneme vocab mismatch: "
                f"{self.settings.phoneme_vocab_path} has size {actual_phoneme_vocab_size}, "
                f"but checkpoint expects {expected_phoneme_vocab_size}"
            )

        _LOGGER.info(
            "Loaded Matcha backend: checkpoint=%s device=%s vocoder=%s sr=%d n_mels=%d",
            self.checkpoint_path,
            self.device,
            self.vocoder_kind,
            self.sample_rate,
            int(self.settings.n_mels),
        )

    def _load_vocoder(self):
        if self.vocoder_kind != "vocos24k":
            raise ValueError(f"Unsupported Starling vocoder: {self.vocoder_kind}")
        from vocos import Vocos  # pylint: disable=import-outside-toplevel

        return Vocos.from_pretrained("charactr/vocos-mel-24khz").eval().to(self.device)

    @staticmethod
    def _normalize_for_alignment(text: str, language: str) -> str:
        from src.piper.preprocess import _map_cld2_to_espeak, _normalize_punct_and_space  # pylint: disable=import-outside-toplevel

        voice = _map_cld2_to_espeak(language, "en-us")
        if voice.lower().startswith("ja"):
            return text
        return " ".join(_normalize_punct_and_space(text).split())

    @staticmethod
    def _flatten_phonemes(sentence) -> str:
        return "".join(sentence)

    def _build_units(self, text: str, language: str, neural: bool) -> tuple[str, list[dict[str, str]], str]:
        from src.piper.heteronym import get_resolver  # pylint: disable=import-outside-toplevel
        from src.piper.preprocess import _map_cld2_to_espeak, _phonemize_espeak_with_mapping  # pylint: disable=import-outside-toplevel

        normalized_text = self._normalize_for_alignment(text, language)
        voice = _map_cld2_to_espeak(language, "en-us")
        sentences, mappings = _phonemize_espeak_with_mapping(normalized_text, voice, None)
        replacements: dict[tuple[int, int], str] = {}

        if neural and voice.lower().startswith("en"):
            for word, h_start, h_end, correct_ipa in get_resolver(device=str(self.device)).resolve_all(normalized_text):
                matched = None
                for sent_idx, sentence_mappings in enumerate(mappings):
                    for word_idx, (text_start, text_len, *_rest) in enumerate(sentence_mappings):
                        map_start = text_start - 1
                        map_end = map_start + text_len
                        if (map_start <= h_start < map_end) or (map_start < h_end <= map_end) or (
                            map_start == h_start and map_end == h_end
                        ):
                            matched = (sent_idx, word_idx)
                            break
                    if matched is not None:
                        break
                if matched is not None:
                    replacements[matched] = correct_ipa
                else:
                    _LOGGER.warning("Matcha neural phonemizer could not map heteronym %r in %r", word, normalized_text)

        units: list[dict[str, str]] = []
        for sent_idx, (sentence, sentence_mappings) in enumerate(zip(sentences, mappings)):
            for word_idx, (text_start, text_len, ph_start, ph_end, punct_len) in enumerate(sentence_mappings):
                if text_start <= 0:
                    continue
                start = text_start - 1
                end = start + text_len
                if punct_len and end < len(normalized_text) and normalized_text[end : end + punct_len].strip():
                    end += punct_len
                unit_text = normalized_text[start:end]
                unit_phonemes = sentence[ph_start:ph_end]
                replacement = replacements.get((sent_idx, word_idx))
                if replacement is not None:
                    word_ph_end = ph_end - punct_len
                    trailing_punct = sentence[word_ph_end:ph_end]
                    unit_phonemes = list(replacement) + trailing_punct
                phonemes = self._flatten_phonemes(unit_phonemes)
                if unit_text and phonemes:
                    units.append({"text": unit_text, "phonemes": phonemes})

        if not units:
            raise ValueError(f"No Matcha alignment units produced for language={language!r}, text={text!r}")

        phoneme_text = " ".join(unit["phonemes"] for unit in units)
        return normalized_text, units, phoneme_text

    def _prepare_item(self, request: MatchaBatchRequest) -> MatchaPreparedItem:
        language = request.language.lower().split("-")[0] or "en"
        normalized_text, units, phoneme_text = self._build_units(request.text, language, request.neural)
        aligned = self.dataset.get_aligned_phoneme_icbpe(phoneme_text, normalized_text, units)
        speaker_id = request.speaker_id if request.speaker_id is not None else 0
        return MatchaPreparedItem(
            text=normalized_text,
            language=language,
            phoneme_text=phoneme_text,
            x_phoneme=aligned["phoneme_ids"],
            x_text_tokens=aligned["text_ids"],
            x_phoneme_mask=aligned["phoneme_mask"],
            x_text_mask=aligned["text_mask"],
            x_unit_ids=aligned["unit_ids"],
            x_unit_texts=aligned["unit_texts"],
            speaker_id=speaker_id,
        )

    def prepare_row(self, row: list[str]) -> MatchaPreparedItem:
        _filepath, lang, phoneme_text, raw_text = row[:4]
        alignment_units = json.loads(row[4])
        aligned = self.dataset.get_aligned_phoneme_icbpe(phoneme_text, raw_text, alignment_units)
        return MatchaPreparedItem(
            text=unicodedata.normalize("NFC", raw_text),
            language=lang,
            phoneme_text=unicodedata.normalize("NFC", phoneme_text),
            x_phoneme=aligned["phoneme_ids"],
            x_text_tokens=aligned["text_ids"],
            x_phoneme_mask=aligned["phoneme_mask"],
            x_text_mask=aligned["text_mask"],
            x_unit_ids=aligned["unit_ids"],
            x_unit_texts=aligned["unit_texts"],
            speaker_id=0,
        )

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
        x_text = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        x_phoneme_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        x_text_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        x_unit_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        x_lengths = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        spks = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        x_unit_texts = []

        for idx, item in enumerate(prepared):
            length = item.x_phoneme.shape[-1]
            x_lengths[idx] = length
            spks[idx] = item.speaker_id
            x[idx, :length] = item.x_phoneme.to(self.device)
            x_text[idx, :length] = item.x_text_tokens.to(self.device)
            x_phoneme_mask[idx, :length] = item.x_phoneme_mask.to(self.device)
            x_text_mask[idx, :length] = item.x_text_mask.to(self.device)
            x_unit_ids[idx, :length] = item.x_unit_ids.to(self.device)
            x_unit_texts.append(item.x_unit_texts)

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
                input_type=input_type,
                x_text=x_text,
                x_text_lengths=x_lengths,
                x_phoneme_mask=x_phoneme_mask,
                x_text_mask=x_text_mask,
                x_unit_ids=x_unit_ids,
                x_unit_texts=x_unit_texts,
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
        prepared = [self._prepare_item(request) for request in requests]
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
                _LOGGER.exception("Matcha batch failed")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
                continue

            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)
