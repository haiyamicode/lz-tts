from __future__ import annotations

import asyncio
import subprocess
import tempfile
import unittest

import numpy as np

from .audio_utils import _audio_to_mp3_bytes
from .take_stream import (
    Mp3Segmenter,
    OrderedChunkAudio,
    StreamingEncodeError,
    StreamingMp3Encoder,
    mp3_duration_seconds,
    parse_mp3_frame,
)

SAMPLE_RATE = 48000


def _tone(seconds: float, frequency: int = 440) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _probe_seconds(audio: bytes) -> float:
    """Decoded length; ffprobe reads no duration from a piped MP3 without a Xing frame."""
    pcm = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "-", "-f", "s16le", "-ac", "1", "-"],
        input=audio,
        capture_output=True,
        check=True,
    ).stdout
    return len(pcm) / 2 / _sample_rate(audio)


def _sample_rate(audio: bytes) -> int:
    for start in range(len(audio) - 3):
        if parse_mp3_frame(audio[start : start + 4]) is not None:
            return (44100, 48000, 32000)[(audio[start + 2] >> 2) & 0x03]
    raise AssertionError("no MP3 frame")


def _ffprobe_seconds(audio: bytes) -> float:
    """What ffprobe makes of the file's own header, which is what the worker used to report."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as handle:
        handle.write(audio)
        handle.flush()
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             handle.name],
            capture_output=True,
            check=True,
        )
    return float(result.stdout.strip())


def _segments(audio: bytes, chunk: int, seconds: float = 1.0):
    segmenter = Mp3Segmenter(seconds)
    segments = []
    for start in range(0, len(audio), chunk):
        segments.extend(segmenter.feed(audio[start : start + chunk]))
    last = segmenter.finish()
    if last is not None:
        segments.append(last)
    return segments


async def _encode(audio: np.ndarray, pieces: int = 7, **adjustments) -> tuple[bytes, bytes]:
    streamed = bytearray()
    encoder = StreamingMp3Encoder(SAMPLE_RATE, on_mp3=streamed.extend, **adjustments)
    for piece in np.array_split(audio, pieces):
        await encoder.write(piece)
    return await encoder.finish(), bytes(streamed)


class Mp3FrameTest(unittest.TestCase):
    def test_parses_mpeg1_layer3(self):
        # 320 kbps, 48 kHz, no padding.
        frame = parse_mp3_frame(bytes([0xFF, 0xFB, 0xE4, 0x00]))
        self.assertEqual(frame.length, 960)
        self.assertAlmostEqual(frame.seconds, 0.024)

    def test_rejects_other_layers_and_junk(self):
        self.assertIsNone(parse_mp3_frame(b"ID3\x04"))
        self.assertIsNone(parse_mp3_frame(bytes([0xFF, 0xFD, 0xE4, 0x00])))  # layer II
        self.assertIsNone(parse_mp3_frame(bytes([0xFF, 0xFB, 0xF4, 0x00])))  # bad bitrate


class Mp3DurationTest(unittest.TestCase):
    def test_counts_a_streamed_take_ffprobe_reads_short(self):
        take, _ = asyncio.run(_encode(_tone(6.0)))
        played = _probe_seconds(take)
        self.assertAlmostEqual(mp3_duration_seconds(take), played, delta=0.05)
        # The take carries no Xing frame, so ffprobe has only the first frame's
        # bitrate to go on and reads the take short - the length that used to be
        # stored, and that the exporter then cut each block to.
        self.assertLess(_ffprobe_seconds(take), played - 0.3)

    def test_counts_a_take_with_an_id3_tag_and_xing_frame(self):
        take = _audio_to_mp3_bytes(_tone(4.0), SAMPLE_RATE)
        self.assertEqual(take[:3], b"ID3")
        played = _probe_seconds(take)
        counted = mp3_duration_seconds(take)
        # Over by the encoder's padding frame, never under: the exporter trims to
        # this, so erring long leaves silence and erring short cuts speech.
        self.assertGreaterEqual(counted, played)
        self.assertAlmostEqual(counted, played, delta=0.05)

    def test_no_frames(self):
        self.assertIsNone(mp3_duration_seconds(b""))
        self.assertIsNone(mp3_duration_seconds(b"not audio"))


class Mp3SegmenterTest(unittest.TestCase):
    def test_segments_join_back_into_the_file(self):
        audio = _audio_to_mp3_bytes(_tone(3.5), SAMPLE_RATE)
        for chunk in (1, 777, len(audio)):
            segments = _segments(audio, chunk)
            self.assertEqual(b"".join(segment.data for segment in segments), audio)
            self.assertEqual(len(segments), 4)
            for segment in segments[:-1]:
                self.assertLessEqual(segment.seconds, 1.0)
                self.assertGreater(segment.seconds, 0.95)
            self.assertAlmostEqual(
                sum(segment.seconds for segment in segments), _probe_seconds(audio), delta=0.06
            )

    def test_each_segment_decodes_on_its_own(self):
        audio = _audio_to_mp3_bytes(_tone(2.5), SAMPLE_RATE)
        for segment in _segments(audio, 4096):
            self.assertAlmostEqual(_probe_seconds(segment.data), segment.seconds, delta=0.05)

    def test_short_take_is_one_segment(self):
        audio = _audio_to_mp3_bytes(_tone(0.3), SAMPLE_RATE)
        segments = _segments(audio, 100)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].data, audio)


class StreamingMp3EncoderTest(unittest.TestCase):
    def test_streams_the_whole_take(self):
        audio, streamed = asyncio.run(_encode(_tone(2.0)))
        self.assertEqual(audio, streamed)
        self.assertAlmostEqual(_probe_seconds(audio), 2.0, delta=0.08)
        self.assertNotEqual(audio[:3], b"ID3")

    def test_applies_adjustments(self):
        audio, _ = asyncio.run(_encode(_tone(2.0), speed=2.0, pause_after=0.5))
        self.assertAlmostEqual(_probe_seconds(audio), 1.5, delta=0.1)

    def test_applies_an_effect_with_a_decode_filter(self):
        audio, _ = asyncio.run(_encode(_tone(1.0), effect="robot"))
        self.assertAlmostEqual(_probe_seconds(audio), 1.0, delta=0.1)

    def test_empty_take_fails(self):
        async def run():
            await StreamingMp3Encoder(SAMPLE_RATE).finish()

        with self.assertRaises(StreamingEncodeError):
            asyncio.run(run())

    def test_abort_stops_the_encoder(self):
        async def run():
            encoder = StreamingMp3Encoder(SAMPLE_RATE, speed=1.5)
            await encoder.write(_tone(0.5))
            await encoder.abort()
            return encoder

        encoder = asyncio.run(run())
        self.assertTrue(all(process.returncode is not None for process in encoder._processes))


class OrderedChunkAudioTest(unittest.TestCase):
    def test_writes_chunks_in_order(self):
        written: list[float] = []

        async def write(audio):
            written.append(float(audio[0]))

        async def run():
            first, second, third = OrderedChunkAudio(write, 3).sinks()
            await first(np.array([0.0]))
            await third(np.array([2.0]))
            await second(np.array([1.0]))
            await third(None)
            await first(np.array([0.5]))
            await first(None)
            await second(np.array([1.5]))
            await second(None)

        asyncio.run(run())
        self.assertEqual(written, [0.0, 0.5, 1.0, 1.5, 2.0])


if __name__ == "__main__":
    unittest.main()
