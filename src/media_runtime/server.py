"""Media runtime: a localhost HTTP process for the music, videogen and lipsync tasks.

Runs from its own venv (src/media_runtime/.venv, Python 3.11, torch 2.14, cu130) so the
YuE2, YingMusic and ComfyUI/H3 dependencies stay apart from lzai. The lzai API
starts it as a child with ``python server.py --port N --features ...``, polls
GET /health until it reports ``ok``, and sends one POST per task.

One model group holds the GPU at a time: YuE2 ("song"), BS-RoFormer +
YingMusic-SVC ("voice"), or the retained H3 ComfyUI executor. A request for
another group unloads the resident one first; requests run one at a time.

GET  /health    -> {"status": "starting" | "ok" | "error", "reason": str, "features": [...]}
POST /generate  -> {"path", "seed", "mode": "prompt" | "clone"}
    body: {"style", "lyrics", "cot"?, "cfg_scale"?, "seed"?, "abc"?, "ode_steps"?,
           "abc_sampling"?, "semantic_sampling"?, "reference_path"?, "output_dir"}
POST /videogen  -> {"path", "seed", "item", "metrics"}
    body: {"job_id", "payload": h3.videogen payload, "output_dir"}
POST /lipsync   -> {"path", "seed", "item", "metrics"}
    body: {"job_id", "payload": h3.lipsync payload, "output_dir"}
"""

from __future__ import annotations

import argparse
import shutil
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import paths  # noqa: F401  (sets the offline Hugging Face cache before model imports)

FEATURES = ("music", "videogen", "lipsync")

app = FastAPI(title="lzai media runtime")
_state: dict = {"status": "starting", "reason": "loading models", "features": []}
_pipeline = None  # MusicPipeline when music is enabled
_gpu_lock = threading.Lock()
_h3_resident = False


class GenerateBody(BaseModel):
    style: str
    lyrics: str
    cot: str = "full"
    cfg_scale: float | None = None
    seed: int | None = None
    abc: str | None = None
    ode_steps: int | None = None
    abc_sampling: dict | None = None
    semantic_sampling: dict | None = None
    reference_path: str | None = None
    output_dir: str


class H3Body(BaseModel):
    job_id: str
    payload: dict[str, Any]
    output_dir: str


@app.get("/health")
def health():
    return dict(_state)


def _require(feature: str) -> None:
    if feature not in _state["features"]:
        raise HTTPException(status_code=404, detail=f"{feature} is not enabled in this runtime")
    if _state["status"] != "ok":
        raise HTTPException(status_code=503, detail=_state["reason"])


def _release_h3() -> None:
    global _h3_resident
    if _h3_resident:
        from h3.minimax_h3_runtime import unload_retained_h3_runtime

        unload_retained_h3_runtime()
        _h3_resident = False


@app.post("/generate")
def generate(body: GenerateBody):
    _require("music")
    from song import SongRequest

    reference = None
    if body.reference_path is not None:
        reference = Path(body.reference_path)
        if not reference.is_file():
            raise HTTPException(status_code=400, detail="reference_path is not a file")
    request = SongRequest(
        style=body.style,
        lyrics=body.lyrics,
        cot=body.cot,
        cfg_scale=body.cfg_scale,
        seed=body.seed,
        abc=body.abc,
        ode_steps=body.ode_steps,
        abc_sampling=body.abc_sampling,
        semantic_sampling=body.semantic_sampling,
    )
    with _gpu_lock:
        _release_h3()
        try:
            result = _pipeline.render(request, Path(body.output_dir), reference_path=reference)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": str(result.path), "seed": result.seed, "mode": result.mode}


def _run_h3(feature: str, body: H3Body) -> dict[str, Any]:
    global _h3_resident
    _require(feature)
    if feature == "videogen":
        from h3.videogen import run_videogen as run
    else:
        from h3.lipsync import run_lipsync_inference as run
    from h3.compat import job_dir

    output_dir = Path(body.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _gpu_lock:
        if _pipeline is not None:
            _pipeline.release()
        _h3_resident = True
        try:
            result = run({"id": body.job_id, "payload": body.payload})
            final = output_dir / f"{feature}-{result['seed']}.mp4"
            shutil.move(result["path"], final)
        except (ValueError, FileNotFoundError) as exc:
            # Bad payload or an input that could not be fetched / read.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(job_dir(body.job_id), ignore_errors=True)
    return {"path": str(final), "seed": result["seed"], "item": result["item"], "metrics": result["metrics"]}


@app.post("/videogen")
def videogen(body: H3Body):
    return _run_h3("videogen", body)


@app.post("/lipsync")
def lipsync(body: H3Body):
    return _run_h3("lipsync", body)


def _check_h3_assets(features: list[str]) -> None:
    from h3 import compat
    from h3.minimax_h3_runtime import (
        ensure_minimax_h3_runtime_assets,
        load_retained_h3_runtime,
        sparse_attention_unavailable,
    )

    global _h3_resident
    comfy_root, _ = ensure_minimax_h3_runtime_assets(generate_audio=True)
    load_retained_h3_runtime(generate_audio=True)
    _h3_resident = True
    if "lipsync" in features:
        from h3.minimax_h3_runtime import LIPSYNC_LORA_NAME

        required = [
            comfy_root / "models" / "loras" / LIPSYNC_LORA_NAME,
            compat.WHISPER_MEDIUM_ROOT / "model.bin",
            compat.SPEECH_CLEANUP_DEMUCS_PATH,
            compat.SPEECH_CLEANUP_ZIPENHANCER_PATH,
            compat.SPEECH_CLEANUP_ZIPENHANCER_CONFIG_PATH,
            compat.SYNCNET_EVAL_SYNCNET_PATH,
            compat.SYNCNET_EVAL_S3FD_PATH,
            compat.GLITCH_DETECTOR_PATH,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("lipsync assets are missing: " + ", ".join(missing))
        reason = sparse_attention_unavailable(comfy_root)
        if reason:
            raise RuntimeError(f"lipsync sparse attention unavailable: {reason}")


def _load(features: list[str], device: str, fp16: bool) -> None:
    global _pipeline
    try:
        if "videogen" in features or "lipsync" in features:
            _check_h3_assets(features)
        if "music" in features:
            from pipeline import MusicPipeline

            _pipeline = MusicPipeline(device=device, fp16=fp16)
            _pipeline.load()
    except Exception as exc:  # reported through /health, never silently dropped
        _state.update(status="error", reason=f"model load failed: {exc}")
        return
    _state.update(status="ok", reason="ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0", help="needs bf16 (Ampere or newer) for YuE2 and H3")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--features", default=",".join(FEATURES), help="comma list of " + ", ".join(FEATURES))
    parser.set_defaults(fp16=True)
    args = parser.parse_args()

    features = [name for name in (part.strip() for part in args.features.split(",")) if name]
    unknown = sorted(set(features) - set(FEATURES))
    if unknown or not features:
        parser.error(f"--features takes a non-empty subset of {FEATURES}, got {args.features!r}")
    _state["features"] = features
    threading.Thread(target=_load, args=(features, args.device, args.fp16), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
