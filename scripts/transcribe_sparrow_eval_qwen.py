#!/usr/bin/env python3
"""Transcribe Sparrow evaluation audio and references with one Qwen3-ASR load."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel


FORCED_LANGUAGES = {"th-TH": "Thai", "pt-BR": "Portuguese"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return "".join(char for char in text if char.isalnum())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--languages",
        default="th-TH,lo-LA,km-KH,my-MM,mn-MN,ps-AF,as-IN,or-IN,he-IL,pt-BR",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    selected = set(filter(None, re.split(r"\s*,\s*", args.languages)))
    rows = [
        row
        for row in json.loads(manifest_path.read_text(encoding="utf-8"))
        if row["language"] in selected
    ]

    requests = []
    for row in rows:
        for kind, path in (("generated", row["path"]), ("reference", row["reference"])):
            requests.append(
                {
                    "kind": kind,
                    "language": row["language"],
                    "source": row["source"],
                    "audio": path,
                    "expected": row["text"],
                }
            )

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map=args.device,
        max_inference_batch_size=4,
        max_new_tokens=256,
    )
    results = model.transcribe(
        audio=[request["audio"] for request in requests],
        language=[FORCED_LANGUAGES.get(request["language"]) for request in requests],
    )

    output = []
    for request, result in zip(requests, results):
        expected = normalize(request["expected"])
        transcript = normalize(result.text or "")
        output.append(
            {
                **request,
                "detected_language": result.language,
                "transcript": (result.text or "").strip(),
                "cer": edit_distance(expected, transcript) / max(1, len(expected)),
            }
        )
        print(
            f"{request['language']} {request['source']} {request['kind']} "
            f"detected={result.language} cer={output[-1]['cer']:.3f} "
            f"text={output[-1]['transcript']}",
            flush=True,
        )

    output_path = Path(args.output) if args.output else manifest_path.with_name("qwen3_asr.json")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
