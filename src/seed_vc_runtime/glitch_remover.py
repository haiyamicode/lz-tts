#!/usr/bin/env python3
"""
Batch safe start-glitch remover (conservative defaults, no fallback trims).

Usage:
  python batch_glitch_remover_safe.py /input/dir /output/dir [--plot] [--dry-run]

Key safety policies:
 - No fallback fixed trimming. If no stable baseline is found -> no cut.
 - Veto: refuse to cut if computed cut_ms > veto_cut_ms (default 50 ms).
 - Dry-run mode: produce only CSV report (no audio files) for validation.

Writes:
 - <basename>_glitch_removed.wav  (only if not dry-run and cut applied or copy saved)
 - glitch_removal_report.csv
"""

import os
import sys
import argparse
import csv
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# ---------- VERY CONSERVATIVE DEFAULTS ----------
DEFAULT_RMS_WINDOW_MS = 5.0  # RMS window (ms)
DEFAULT_RMS_HOP_MS = 1.0  # RMS hop (ms)
DEFAULT_RMS_THR = 0.001  # lower threshold = conservative
DEFAULT_HOLD_MS = 30.0  # must remain below thr for 30 ms
DEFAULT_SAFETY_MS = 2.0  # keep 2 ms before baseline
DEFAULT_MAX_CUT_MS = 50.0  # never cut more than 50 ms
DEFAULT_FALLBACK_TRIM_MS = 0.0  # DISABLED: do not fallback to fixed trim
DEFAULT_VETO_CUT_MS = 50.0  # if cut > this, refuse to cut
SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


# ---------- Helpers ----------
def load_to_wav_array(path):
    audio = AudioSegment.from_file(path)
    tmp_wav = path + ".tmp_for_read.wav"
    audio.export(tmp_wav, format="wav")
    x, sr = sf.read(tmp_wav)
    os.remove(tmp_wav)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    # normalize int -> float
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1.0)
    else:
        x = x.astype(np.float32)
    return x, sr


def rms_envelope(x, sr, win_ms=5.0, hop_ms=1.0):
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    rms = []
    pos = []
    # make sure we can compute at least one frame
    end = max(1, len(x) - win + 1)
    for s in range(0, end, hop):
        frame = x[s : s + win]
        rms.append(np.sqrt(np.mean(frame * frame) + 1e-18))
        pos.append(s)
    return np.array(rms), np.array(pos), win, hop


def find_stable_baseline(rms, pos, thr, hold_ms, hop_ms):
    hold_frames = max(1, int(np.ceil(hold_ms / hop_ms)))
    for i in range(len(rms)):
        if rms[i] < thr:
            end_idx = i + hold_frames
            if end_idx <= len(rms) and np.all(rms[i:end_idx] < thr):
                return pos[i]
    return None


def process_file(path, out_dir, params, do_write):
    x, sr = load_to_wav_array(path)
    rms, pos, win, hop = rms_envelope(
        x, sr, win_ms=params["rms_win_ms"], hop_ms=params["rms_hop_ms"]
    )
    stable_sample = find_stable_baseline(
        rms, pos, params["rms_thr"], params["hold_ms"], params["rms_hop_ms"]
    )

    safety_samples = int(sr * params["safety_ms"] / 1000.0)
    max_cut_samples = int(sr * params["max_cut_ms"] / 1000.0)
    veto_cut_samples = int(sr * params["veto_cut_ms"] / 1000.0)

    if stable_sample is not None:
        cut_sample = max(0, stable_sample - safety_samples)
        cut_sample = min(cut_sample, max_cut_samples)
        cut_ms = round(1000.0 * cut_sample / sr, 3)
        # veto if cut exceeds veto threshold
        if cut_sample > veto_cut_samples:
            decision = f"vetoed_cut (would_cut={cut_ms}ms > veto {params['veto_cut_ms']}ms)"
            cut_sample = 0
        else:
            decision = f"baseline_detected - cut {cut_ms} ms"
    else:
        # No baseline found => NO CUT (conservative)
        cut_sample = 0
        decision = "no_baseline_detected - no_cut"

    out = x[cut_sample:]
    base = os.path.splitext(os.path.basename(path))[0]
    out_fn = os.path.join(out_dir, f"{base}_glitch_removed.wav")

    if do_write:
        # always write output copy (either trimmed or original) to allow inspection
        sf.write(out_fn, out, sr, subtype="PCM_24")

    return {
        "src": path,
        "out": out_fn,
        "sr": sr,
        "cut_samples": int(cut_sample),
        "cut_ms": round(1000.0 * cut_sample / sr, 3),
        "decision": decision,
    }


# ---------- CLI ----------
def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Batch safe start-glitch remover (conservative defaults)."
    )
    p.add_argument("input_dir", help="Folder with audio files (or single audio file).")
    p.add_argument("output_dir", help="Folder where outputs and report.csv are saved.")
    p.add_argument("--rms-win-ms", type=float, default=DEFAULT_RMS_WINDOW_MS)
    p.add_argument("--rms-hop-ms", type=float, default=DEFAULT_RMS_HOP_MS)
    p.add_argument("--rms-thr", type=float, default=DEFAULT_RMS_THR)
    p.add_argument("--hold-ms", type=float, default=DEFAULT_HOLD_MS)
    p.add_argument("--safety-ms", type=float, default=DEFAULT_SAFETY_MS)
    p.add_argument("--max-cut-ms", type=float, default=DEFAULT_MAX_CUT_MS)
    p.add_argument(
        "--veto-cut-ms",
        type=float,
        default=DEFAULT_VETO_CUT_MS,
        help="If computed cut > veto-cut-ms, refuse to cut (safest).",
    )
    p.add_argument(
        "--fallback-trim-ms",
        type=float,
        default=DEFAULT_FALLBACK_TRIM_MS,
        help="DISABLED by default. Never use fixed fallback trimming.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Do not write audio files; only produce CSV report."
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Optional: save start-of-file plots (requires matplotlib).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    IN = args.input_dir
    OUT = args.output_dir
    os.makedirs(OUT, exist_ok=True)

    params = {
        "rms_win_ms": args.rms_win_ms,
        "rms_hop_ms": args.rms_hop_ms,
        "rms_thr": args.rms_thr,
        "hold_ms": args.hold_ms,
        "safety_ms": args.safety_ms,
        "max_cut_ms": args.max_cut_ms,
        "veto_cut_ms": args.veto_cut_ms,
        "fallback_trim_ms": args.fallback_trim_ms,
    }

    # collect files
    files = []
    if os.path.isfile(IN) and IN.lower().endswith(SUPPORTED_EXTS):
        files = [IN]
    elif os.path.isdir(IN):
        for fn in sorted(os.listdir(IN)):
            if fn.lower().endswith(SUPPORTED_EXTS):
                files.append(os.path.join(IN, fn))
    else:
        print("No supported files found at input path.")
        sys.exit(1)

    results = []
    for f in files:
        print("Processing:", f)
        try:
            r = process_file(f, OUT, params, do_write=not args.dry_run)
            print(" ->", r["decision"], f"({r['cut_ms']} ms)")
            results.append(r)
        except Exception as e:
            print(" ERROR processing", f, ":", e)

    # write CSV report
    csv_path = os.path.join(OUT, "glitch_removal_report.csv")
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["source", "output", "sample_rate", "cut_samples", "cut_ms", "decision"])
        for r in results:
            w.writerow([r["src"], r["out"], r["sr"], r["cut_samples"], r["cut_ms"], r["decision"]])
    print("Done. Report:", csv_path)
    if args.dry_run:
        print(
            "Dry-run: no audio files were written. Inspect the CSV and re-run without --dry-run to write outputs."
        )


if __name__ == "__main__":
    main()
