#!/usr/bin/env python3
"""Grow one chat to the Gemma 4 context limit and record llama.cpp timings."""

import argparse
import hashlib
import json
import math
import pathlib
import time
import urllib.request


TARGETS = (2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 196608, 245760)
FILLER = (
    "Evidence about consciousness, mortality, responsibility, meaning, and human "
    "flourishing should be weighed without assuming the conclusion. "
)


def post(endpoint, path, payload):
    request = urllib.request.Request(
        endpoint + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        },
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response), time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    filler_tokens = len(post(args.endpoint, "/tokenize", {"content": FILLER})[0]["tokens"])
    messages = []
    turns = []
    previous_prompt_tokens = 0
    for turn, target in enumerate(TARGETS, 1):
        wanted = max(64, target - previous_prompt_tokens - 64)
        content = (
            f"Debate turn {turn}: continue examining the meaning of life. "
            "Synthesize the evidence below, challenge your prior answer, and reply "
            "in no more than two sentences.\n\n"
            + FILLER * math.ceil(wanted / filler_tokens)
        )
        messages.append({"role": "user", "content": content})
        response, wall_s = post(
            args.endpoint,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        answer = response["choices"][0]["message"]["content"]
        if not answer:
            raise RuntimeError(f"turn {turn} returned no visible content")
        messages.append({"role": "assistant", "content": answer})
        usage = response["usage"]
        previous_prompt_tokens = usage["prompt_tokens"]
        result = {
            "turn": turn,
            "target_prompt_tokens": target,
            "input_chars": len(content),
            "wall_s": round(wall_s, 3),
            "usage": usage,
            "timings": response.get("timings"),
            "finish_reason": response["choices"][0].get("finish_reason"),
            "response_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "response_preview": answer[:160],
        }
        turns.append(result)
        print(
            f"turn={turn} prompt={previous_prompt_tokens} "
            f"cached={usage.get('prompt_tokens_details', {}).get('cached_tokens')} "
            f"input_tps={result['timings'].get('prompt_per_second')} "
            f"output_tps={result['timings'].get('predicted_per_second')}",
            flush=True,
        )

    run = {
        "started_at": started_at,
        "model": args.model,
        "targets": TARGETS,
        "turns": turns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output:
        output.write(json.dumps(run, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
