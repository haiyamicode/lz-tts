"""FastAPI-based TTS API server."""

__all__ = ["app", "create_app", "run"]


def __getattr__(name: str):
    if name in __all__:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
