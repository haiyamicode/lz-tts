"""Media runtime: the child process that runs the music, videogen, lipsync and imagegen tasks.

Runs from its own venv (src/media_runtime/.venv, Python 3.11, torch 2.14, cu130) so the
YuE2, YingMusic and ComfyUI/H3 dependencies stay apart from lzai — the one reason this is a
process of its own and not a worker thread. The lzai API spawns it with a channel of two pipes
(``channel.py``), waits for the ``state`` message its models are loaded, and sends one request
per task. Nothing listens on a port; the channel ends with either process.

One model group holds the GPU at a time: YuE2 ("song"), BS-RoFormer +
YingMusic-SVC ("voice"), or H3 (the retained ComfyUI executor + the helper
models its jobs keep: ZipEnhancer, CTC, SyncNet/S3FD, glitch, RIFE). Videogen,
lipsync and imagegen share the H3 group, so nothing unloads between them; only
a music job releases it. Requests run one at a time.

state (sent, not asked for) -> {"status": "starting" | "ok" | "error", "reason": str, "features": [...]}
generate  -> {"path", "seed", "mode": "prompt" | "clone"}
    body: {"style", "lyrics", "cot"?, "cfg_scale"?, "seed"?, "abc"?, "ode_steps"?,
           "abc_sampling"?, "semantic_sampling"?, "reference_path"?, "output_dir"}
videogen  -> {"path", "seed", "item", "metrics", "last_frame_path"?}
    body: {"job_id", "payload": h3.videogen payload, "output_dir"}
lipsync   -> {"path", "seed", "item", "metrics", "files"?: {"still<k>": path}}
    body: {"job_id", "payload": h3.lipsync payload, "output_dir"}
imagegen  -> {"path", "seed", "item", "metrics"}  (a PNG)
    body: {"job_id", "payload": h3.imagegen payload, "output_dir"}
cancel    -> {"running": bool}  (never waits for the GPU)
    body: {"job_id"}: that job stops at its next check (``cancel.py``) and is answered with a
    ``cancelled`` error; a job not started yet stops as it starts.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

import cancel
import channel
import paths  # noqa: F401  (sets the offline Hugging Face cache before model imports)
import progress
from channel import RequestError

FEATURES = ("music", "videogen", "lipsync", "imagegen")
H3_FEATURES = ("videogen", "lipsync", "imagegen")

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
    #: What the job's progress is reported under; without one it is not reported.
    job_id: str | None = None


class H3Body(BaseModel):
    job_id: str
    payload: dict[str, Any]
    output_dir: str


class CancelBody(BaseModel):
    job_id: str


def cancel_job(body: CancelBody) -> dict[str, Any]:
    return {"running": cancel.ask(body.job_id)}


def _require(feature: str) -> None:
    if feature not in _state["features"]:
        raise RequestError(f"{feature} is not enabled in this runtime")
    if _state["status"] != "ok":
        raise RequestError(_state["reason"], kind="runtime")


def _release_h3() -> None:
    """Give the GPU to the music models: drop the H3 executor and the helper models the H3 jobs keep resident
    (ZipEnhancer, the CTC transcriber, SyncNet/S3FD, the glitch detector, RIFE). Only a music job calls it."""
    global _h3_resident
    if _h3_resident:
        import gc

        import torch

        from h3 import h3_interpolation, lipsync_checks, speech_chunks, speech_cleanup
        from h3.minimax_h3_runtime import unload_retained_h3_runtime

        unload_retained_h3_runtime()
        speech_cleanup._ZIPENHANCER = None
        speech_chunks._CTC.clear()
        lipsync_checks._MODELS.clear()
        lipsync_checks._GLITCH_NET.clear()
        h3_interpolation._RIFE = None
        gc.collect()
        torch.cuda.empty_cache()
        _h3_resident = False


def generate(body: GenerateBody) -> dict[str, Any]:
    _require("music")
    from song import SongRequest

    reference = None
    if body.reference_path is not None:
        reference = Path(body.reference_path)
        if not reference.is_file():
            raise RequestError("reference_path is not a file")
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
    with _gpu_lock, progress.job(body.job_id), _cancellable(body.job_id):
        _release_h3()
        try:
            result = _pipeline.render(request, Path(body.output_dir), reference_path=reference)
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
    return {"path": str(result.path), "seed": result.seed, "mode": result.mode}


@contextlib.contextmanager
def _cancellable(job_id: str | None) -> Iterator[None]:
    """Runs ``job_id`` under :mod:`cancel`; a cancelled job is answered as a ``cancelled`` error.

    Whatever a cancelled job fails with counts as the cancel (YuE2's hook raises ``InterruptedError``,
    an interrupted ComfyUI graph only fails).
    """
    try:
        with cancel.job(job_id):
            try:
                yield
            except BaseException as exc:
                if cancel.is_cancelled() and not isinstance(exc, cancel.Cancelled):
                    raise cancel.Cancelled(f"job {job_id} was cancelled") from exc
                raise
    except cancel.Cancelled as exc:
        raise RequestError(str(exc), kind="cancelled") from exc


def _run_h3(feature: str, body: H3Body) -> dict[str, Any]:
    global _h3_resident
    _require(feature)
    if feature == "videogen":
        from h3.videogen import run_videogen as run
    elif feature == "imagegen":
        from h3.imagegen import run_imagegen as run
    else:
        from h3.lipsync import run_lipsync_inference as run
    from h3.compat import job_dir

    output_dir = Path(body.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _gpu_lock, progress.job(body.job_id), _cancellable(body.job_id):
        if _pipeline is not None:
            _pipeline.release()
        _h3_resident = True
        try:
            result = run({"id": body.job_id, "payload": body.payload})
            progress.report("finishing")
            final = output_dir / f"{feature}-{result['seed']}{Path(result['path']).suffix}"
            shutil.move(result["path"], final)
            last_frame = None
            if result.get("last_frame_path"):
                # A continuable videogen clip's restored last frame: the next job opens on it.
                last_frame = output_dir / f"{feature}-{result['seed']}-last-frame{Path(result['last_frame_path']).suffix}"
                shutil.move(result["last_frame_path"], last_frame)
            # Further result files by name (a generate lipsync's rest stills), out of the job's scratch.
            files = {}
            for name, path in (result.get("files") or {}).items():
                files[name] = str(output_dir / f"{feature}-{result['seed']}-{name}{Path(path).suffix}")
                shutil.move(path, files[name])
        except (ValueError, FileNotFoundError) as exc:
            # Bad payload or an input that could not be fetched / read.
            raise RequestError(str(exc)) from exc
        finally:
            shutil.rmtree(job_dir(body.job_id), ignore_errors=True)
    return {
        "path": str(final),
        "seed": result["seed"],
        "item": result["item"],
        "metrics": result["metrics"],
        **({"last_frame_path": str(last_frame)} if last_frame else {}),
        **({"files": files} if files else {}),
    }


def _operation(model: type[BaseModel], run) -> channel.Operation:
    """One operation of the channel: its body validated, then run."""

    def operation(body: dict[str, Any]) -> dict[str, Any]:
        try:
            validated = model.model_validate(body)
        except ValidationError as error:
            raise RequestError(str(error)) from error
        return run(validated)

    return operation


OPERATIONS: dict[str, channel.Operation] = {
    "generate": _operation(GenerateBody, generate),
    "videogen": _operation(H3Body, lambda body: _run_h3("videogen", body)),
    "lipsync": _operation(H3Body, lambda body: _run_h3("lipsync", body)),
    "imagegen": _operation(H3Body, lambda body: _run_h3("imagegen", body)),
    "cancel": _operation(CancelBody, cancel_job),
}


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
    # Every preset of every enabled op must be servable.
    from h3.h3_still import STILL_PRESETS
    from h3.minimax_h3_runtime import LIPSYNC_PRESETS, VIDEO_ADDON_LORAS, VIDEO_PRESETS

    lora_names = set()
    if "imagegen" in features:
        lora_names |= {spec["lora"]["name"] for spec in STILL_PRESETS.values() if spec["lora"]}
    if "videogen" in features:
        lora_names |= {recipe.lora_name for recipe in VIDEO_PRESETS.values() if recipe.lora_name}
        lora_names |= set(VIDEO_ADDON_LORAS.values())
    if "lipsync" in features:
        lora_names |= {recipe.lora_name for recipe in LIPSYNC_PRESETS.values() if recipe.lora_name}
    missing = [str(path) for path in (comfy_root / "models" / "loras" / name for name in sorted(lora_names)) if not path.exists()]
    if missing:
        raise FileNotFoundError("H3 preset weights are missing: " + ", ".join(missing))
    # Rest stills (imagegen rest_frame, lipsync pinned_frames generate) check the face of every take.
    if "imagegen" in features or "lipsync" in features:
        realism = compat.REALISM_CLIP_DIR / "model.safetensors"
        for path in (compat.FACE_LANDMARKER_PATH, compat.FACE_IDENTITY_PATH, realism):
            if not path.exists():
                raise FileNotFoundError(f"the rest still face model is missing: {path}")
    if ("videogen" in features or "lipsync" in features) and not compat.SPAN_UPSCALE_MODEL_PATH.exists():
        raise FileNotFoundError(f"the SPAN upscale model is missing: {compat.SPAN_UPSCALE_MODEL_PATH}")
    if "lipsync" in features:
        required = [
            compat.CTC_ROOT / "model.safetensors",
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
        if any(feature in H3_FEATURES for feature in features):
            _check_h3_assets(features)
        if "music" in features:
            from pipeline import MusicPipeline

            _pipeline = MusicPipeline(device=device, fp16=fp16)
            _pipeline.load()
    except Exception as exc:  # told to the parent, never silently dropped
        _state.update(status="error", reason=f"model load failed: {exc}")
        channel.send({"state": dict(_state)})
        return
    _state.update(status="ok", reason="ready")
    channel.send({"state": dict(_state)})


def main() -> None:
    parser = argparse.ArgumentParser()
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
    # The channel first: a load that finishes before the first request still has somewhere to report.
    channel.connect()
    threading.Thread(target=_load, args=(features, args.device, args.fp16), daemon=True).start()
    channel.serve(OPERATIONS)


if __name__ == "__main__":
    main()
