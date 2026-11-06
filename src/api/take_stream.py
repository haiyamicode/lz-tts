"""Encode a take while it is generated and cut the MP3 into ~1 s segments.

The take's MP3 is one continuous LAME stream, so its segments join back into the
exact file a finished encode would give, and each segment plays on its own as
an HLS packed-audio segment.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np

from .audio_adjustments import (
    AudioAdjustmentError,
    AudioEffect,
    decode_filter,
    needs_adjustment,
    sox_adjustment_effects,
)
from .audio_utils import MP3_BITRATE, MP3_EXPORT_PARAMETERS, _audio_to_float32


_LOGGER = logging.getLogger(__name__)

# Rate the adjustment pass ends at, as sox_adjustment_effects does.
_ADJUSTED_SAMPLE_RATE = 44100

_MPEG1_BITRATES = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_MPEG2_BITRATES = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
_SAMPLE_RATES = {
    3: (44100, 48000, 32000),  # MPEG 1
    2: (22050, 24000, 16000),  # MPEG 2
    0: (11025, 12000, 8000),  # MPEG 2.5
}


@dataclass(frozen=True)
class Mp3Frame:
    length: int
    seconds: float


def parse_mp3_frame(header: bytes | bytearray | memoryview) -> Mp3Frame | None:
    """The Layer III frame a 4-byte header starts, or None when it isn't one."""
    if len(header) < 4 or header[0] != 0xFF or header[1] & 0xE0 != 0xE0:
        return None
    version = (header[1] >> 3) & 0x03
    layer = (header[1] >> 1) & 0x03
    bitrate_index = header[2] >> 4
    rate_index = (header[2] >> 2) & 0x03
    if version == 1 or layer != 1 or bitrate_index in (0, 15) or rate_index == 3:
        return None
    padding = (header[2] >> 1) & 0x01
    sample_rate = _SAMPLE_RATES[version][rate_index]
    if version == 3:
        bitrate = _MPEG1_BITRATES[bitrate_index] * 1000
        return Mp3Frame(144 * bitrate // sample_rate + padding, 1152 / sample_rate)
    bitrate = _MPEG2_BITRATES[bitrate_index] * 1000
    return Mp3Frame(72 * bitrate // sample_rate + padding, 576 / sample_rate)


def _id3v2_length(data: bytes | bytearray) -> int | None:
    """Length of the ID3v2 tag ``data`` starts with, or None if it needs more bytes."""
    if len(data) < 10:
        return None
    size = 0
    for byte in data[6:10]:
        size = (size << 7) | (byte & 0x7F)
    footer = 10 if data[5] & 0x10 else 0
    return 10 + size + footer


@dataclass(frozen=True)
class Mp3Segment:
    data: bytes
    seconds: float


class Mp3Segmenter:
    """Cuts an MP3 stream at frame boundaries into segments of at most ``seconds``.

    Bytes that aren't frames (an ID3 tag, junk before a sync) stay with the
    segment they fall in and add no duration.
    """

    def __init__(self, seconds: float = 1.0):
        self.seconds = seconds
        self._buffer = bytearray()
        # Where the next unparsed frame starts, and the duration before it.
        self._parsed = 0
        self._duration = 0.0
        self._started = False

    def feed(self, data: bytes) -> list[Mp3Segment]:
        self._buffer.extend(data)
        segments: list[Mp3Segment] = []
        buffer = self._buffer
        if not self._started:
            if len(buffer) < 3:
                return segments
            if buffer[:3] == b"ID3":
                tag_length = _id3v2_length(buffer)
                if tag_length is None:
                    return segments
                self._parsed = tag_length
            self._started = True
        while self._parsed + 4 <= len(buffer):
            frame = parse_mp3_frame(buffer[self._parsed : self._parsed + 4])
            if frame is None:
                self._parsed += 1
                continue
            if self._parsed + frame.length > len(buffer):
                break
            if self._duration and self._duration + frame.seconds > self.seconds + 1e-9:
                segments.append(Mp3Segment(bytes(buffer[: self._parsed]), self._duration))
                del buffer[: self._parsed]
                self._parsed = 0
                self._duration = 0.0
            self._parsed += frame.length
            self._duration += frame.seconds
        return segments

    def finish(self) -> Mp3Segment | None:
        """The rest of the stream, as the last segment."""
        if not self._buffer:
            return None
        segment = Mp3Segment(bytes(self._buffer), self._duration)
        self._buffer.clear()
        self._parsed = 0
        self._duration = 0.0
        return segment


class StreamingEncodeError(AudioAdjustmentError):
    pass


class StreamingMp3Encoder:
    """Encodes a take's audio to MP3 as it arrives, with its adjustments applied.

    Without adjustments the take is encoded at its own rate, as
    ``_audio_to_mp3_bytes`` does; with them it runs the same ffmpeg filter and
    SoX chain as ``adjust_audio``, ending at 44.1 kHz. ``on_mp3`` gets the MP3
    bytes as the encoder writes them.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        on_mp3: Callable[[bytes], None] | None = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        pause_before: float = 0.0,
        pause_after: float = 0.0,
        effect: AudioEffect | None = None,
    ):
        self.sample_rate = sample_rate
        self._on_mp3 = on_mp3
        self._adjustments = dict(
            speed=speed,
            pitch=pitch,
            volume=volume,
            pause_before=pause_before,
            pause_after=pause_after,
            effect=effect,
        )
        self._processes: list[asyncio.subprocess.Process] = []
        self._stderr_tasks: list[asyncio.Task[bytes]] = []
        self._reader: asyncio.Task[None] | None = None
        self._output = bytearray()
        self._error: BaseException | None = None
        self._started = False

    def _commands(self) -> list[list[str]]:
        rate = self.sample_rate
        commands: list[list[str]] = []
        if needs_adjustment(**self._adjustments):
            filter_steps = decode_filter(self._adjustments["effect"])
            if filter_steps:
                commands.append(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0",
                        "-af", filter_steps, "-ar", str(_ADJUSTED_SAMPLE_RATE),
                        "-flush_packets", "1", "-f", "f32le", "pipe:1",
                    ]
                )
                rate = _ADJUSTED_SAMPLE_RATE
            commands.append(
                [
                    "sox", "-V1", "--buffer", "4096",
                    "-t", "f32", "-r", str(rate), "-c", "1", "-",
                    "-t", "f32", "-",
                    *sox_adjustment_effects(**self._adjustments),
                ]
            )
            rate = _ADJUSTED_SAMPLE_RATE
        commands.append(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0",
                "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, *MP3_EXPORT_PARAMETERS,
                "-id3v2_version", "0", "-write_xing", "0",
                "-flush_packets", "1", "-f", "mp3", "pipe:1",
            ]
        )
        return commands

    async def _start(self) -> None:
        self._started = True
        commands = self._commands()
        stdin: int = asyncio.subprocess.PIPE
        for index, command in enumerate(commands):
            last = index == len(commands) - 1
            read_fd, write_fd = (None, None) if last else os.pipe()
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=stdin,
                    stdout=asyncio.subprocess.PIPE if last else write_fd,
                    stderr=asyncio.subprocess.PIPE,
                )
            finally:
                if stdin != asyncio.subprocess.PIPE:
                    os.close(stdin)
                if write_fd is not None:
                    os.close(write_fd)
            self._processes.append(process)
            assert process.stderr is not None
            self._stderr_tasks.append(asyncio.create_task(process.stderr.read()))
            if read_fd is not None:
                stdin = read_fd
        self._reader = asyncio.create_task(self._read_output(self._processes[-1]))

    async def _read_output(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        while chunk := await process.stdout.read(65536):
            self._output.extend(chunk)
            if self._on_mp3 is not None:
                try:
                    self._on_mp3(chunk)
                except Exception:
                    _LOGGER.exception("Streaming MP3 consumer failed")

    async def write(self, audio: np.ndarray) -> None:
        """Feeds the next audio. Never raises: a failure surfaces at ``finish``."""
        if self._error is not None or not len(audio):
            return
        try:
            if not self._started:
                await self._start()
            stdin = self._processes[0].stdin
            assert stdin is not None
            stdin.write(_audio_to_float32(audio).tobytes())
            await stdin.drain()
        except Exception as error:
            self._error = error

    async def finish(self) -> bytes:
        """Ends the stream and returns the whole MP3."""
        if self._error is None and not self._started:
            raise StreamingEncodeError("cannot encode empty audio")
        if self._error is None:
            stdin = self._processes[0].stdin
            assert stdin is not None
            try:
                stdin.close()
                await stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError) as error:
                self._error = error
        if self._error is not None:
            await self.abort()
            raise StreamingEncodeError(f"streaming encode failed: {self._error}") from self._error
        assert self._reader is not None
        await self._reader
        for process, stderr_task in zip(self._processes, self._stderr_tasks):
            code = await process.wait()
            stderr = (await stderr_task).decode(errors="replace").strip()
            if code != 0:
                raise StreamingEncodeError(
                    f"streaming encode failed with exit code {code}: {stderr[:1000]}"
                )
        return bytes(self._output)

    async def abort(self) -> None:
        for process in self._processes:
            if process.returncode is None:
                process.kill()
        for process in self._processes:
            await process.wait()
        for task in (*self._stderr_tasks, *([self._reader] if self._reader else [])):
            task.cancel()
        await asyncio.gather(
            *self._stderr_tasks, *([self._reader] if self._reader else []),
            return_exceptions=True,
        )


class OrderedChunkAudio:
    """Joins the live audio of a line spoken as several chunks, in chunk order.

    Chunks generate concurrently; the first one goes straight through, later
    ones are held until the chunks before them are complete. ``sinks()`` gives
    one ``AudioSink`` per chunk.
    """

    def __init__(self, write: Callable[[np.ndarray], Awaitable[None]], chunk_count: int):
        self._write = write
        self._pending: list[list[np.ndarray]] = [[] for _ in range(chunk_count)]
        self._complete = [False] * chunk_count
        self._current = 0
        self._lock = asyncio.Lock()

    def sinks(self) -> list[Callable[[np.ndarray | None], Awaitable[None]]]:
        return [self._sink(index) for index in range(len(self._pending))]

    def _sink(self, index: int) -> Callable[[np.ndarray | None], Awaitable[None]]:
        async def sink(audio: np.ndarray | None) -> None:
            async with self._lock:
                if audio is None:
                    self._complete[index] = True
                elif index == self._current:
                    await self._write(audio)
                    return
                else:
                    self._pending[index].append(audio)
                    return
                while self._current < len(self._pending) and self._complete[self._current]:
                    self._current += 1
                    if self._current < len(self._pending):
                        for held in self._pending[self._current]:
                            await self._write(held)
                        self._pending[self._current].clear()

        return sink
