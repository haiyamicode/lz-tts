"""Music runtime: a localhost HTTP process for the generate-music task.

Runs from its own venv (src/music_runtime/.venv, Python 3.11, torch 2.10) so the
ACE-Step and YingMusic dependencies stay apart from lz-tts. The lz-tts API
starts it as a child with ``python server.py --port N``, polls GET /health until
it reports ``ok``, and sends POST /generate for each task.

GET  /health    -> {"status": "starting" | "ok" | "error", "reason": str}
POST /generate  -> {"path", "seed", "mode": "prompt" | "clone"}
    body: {"caption", "duration", "lyrics"?, "instrumental"?, "vocal_language"?,
           "bpm"?, "keyscale"?, "timesignature"?, "seed"?, "inference_steps"?,
           "reference_path"?, "output_dir"}
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import paths  # noqa: F401  (sets the offline Hugging Face cache before model imports)
from pipeline import MusicPipeline
from song import SongRequest

app = FastAPI(title="lzai music runtime")
_state: dict = {"status": "starting", "reason": "loading models"}
_pipeline: MusicPipeline | None = None


class GenerateBody(BaseModel):
    caption: str
    duration: float
    lyrics: str = ""
    instrumental: bool = False
    vocal_language: str = "en"
    bpm: int | None = None
    keyscale: str | None = None
    timesignature: str = "4/4"
    seed: int | None = None
    inference_steps: int = 8
    reference_path: str | None = None
    output_dir: str


@app.get("/health")
def health():
    return dict(_state)


@app.post("/generate")
def generate(body: GenerateBody):
    if _state["status"] != "ok" or _pipeline is None:
        raise HTTPException(status_code=503, detail=_state["reason"])
    reference = None
    if body.reference_path is not None:
        reference = Path(body.reference_path)
        if not reference.is_file():
            raise HTTPException(status_code=400, detail="reference_path is not a file")
    request = SongRequest(
        caption=body.caption,
        duration=body.duration,
        lyrics=body.lyrics,
        instrumental=body.instrumental,
        vocal_language=body.vocal_language,
        bpm=body.bpm,
        keyscale=body.keyscale,
        timesignature=body.timesignature,
        seed=body.seed,
        inference_steps=body.inference_steps,
    )
    try:
        result = _pipeline.render(request, Path(body.output_dir), reference_path=reference)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": str(result.path), "seed": result.seed, "mode": result.mode}


def _load(pipeline: MusicPipeline) -> None:
    try:
        pipeline.load()
    except Exception as exc:  # reported through /health, never silently dropped
        _state.update(status="error", reason=f"model load failed: {exc}")
        return
    _state.update(status="ok", reason="ready")


def main() -> None:
    global _pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0", help="needs bf16 (Ampere or newer) for ACE-Step")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=True)
    args = parser.parse_args()

    _pipeline = MusicPipeline(device=args.device, fp16=args.fp16)
    threading.Thread(target=_load, args=(_pipeline,), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
