"""Piper TTS inference module."""

__all__ = ["PiperInference"]


def __getattr__(name):
    if name == "PiperInference":
        from .inference import PiperInference

        return PiperInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
