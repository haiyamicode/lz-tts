import typing
from pathlib import Path

import numpy as np
import onnxruntime


class SileroVoiceActivityDetector:
    """Detects speech/silence using Silero VAD.

    https://github.com/snakers4/silero-vad
    """

    def __init__(
        self,
        onnx_path: typing.Union[str, Path],
        providers: typing.Optional[typing.Sequence[str]] = None,
    ):
        onnx_path = str(onnx_path)

        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_providers = None
        if providers is not None:
            available = set(onnxruntime.get_available_providers())
            session_providers = [provider for provider in providers if provider in available]
            if not session_providers:
                raise RuntimeError(
                    f"None of the requested Silero VAD ONNX providers are available: {providers}. "
                    f"Available providers: {sorted(available)}."
                )

        self.session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=session_providers,
        )
        self.providers = self.session.get_providers()

        self._h = np.zeros((2, 1, 64)).astype("float32")
        self._c = np.zeros((2, 1, 64)).astype("float32")

    def reset(self):
        self._h.fill(0.0)
        self._c.fill(0.0)

    def __call__(self, audio_array: np.ndarray, sample_rate: int = 16000):
        """Return probability of speech in audio [0-1].

        Audio must be 16Khz 16-bit mono PCM.
        """
        if len(audio_array.shape) == 1:
            # Add batch dimension
            audio_array = np.expand_dims(audio_array, 0)

        if len(audio_array.shape) > 2:
            raise ValueError(
                f"Too many dimensions for input audio chunk {audio_array.shape}"
            )

        if audio_array.shape[0] > 1:
            raise ValueError("Onnx model does not support batching")

        if sample_rate != 16000:
            raise ValueError("Only 16Khz audio is supported")

        ort_inputs = {
            "input": audio_array.astype(np.float32),
            "h0": self._h,
            "c0": self._c,
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs

        out = out.squeeze(2)[:, 1]  # make output type match JIT analog

        return out
