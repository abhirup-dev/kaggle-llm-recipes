#!/usr/bin/env python3
"""Run a repeatable OpenAI-compatible llama.cpp benchmark over a tailnet."""

import argparse
import hashlib
import json
import pathlib
import time
import urllib.request


CASES = [
    (
        "exact",
        "Reply with exactly: benchmark-ready",
        32,
    ),
    (
        "short_qa",
        "What is the difference between RAM and VRAM? Answer in two sentences.",
        128,
    ),
    (
        "arithmetic",
        "A notebook downloads 3.0 GB in 18.4 seconds and loads the model in "
        "another 72.2 seconds. Calculate download throughput in MB/s and total "
        "time, showing the calculation.",
        192,
    ),
    (
        "long_input",
        (
            "Kaggle notebooks provide ephemeral GPU compute, local temporary "
            "storage, internet access, and attached read-only datasets. "
        )
        * 180
        + "Summarize the operational tradeoffs in exactly three bullets.",
        96,
    ),
    (
        "generation",
        "Explain mixture-of-experts inference in about 200 words, focusing on "
        "active parameters, memory footprint, quantization, and throughput.",
        320,
    ),
]


def request_json(url, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        },
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), time.monotonic() - started


def completion(endpoint, model, name, prompt, max_tokens):
    response, wall_s = request_json(
        endpoint + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    message = response["choices"][0]["message"]
    text = message["content"]
    reasoning = message.get("reasoning_content") or ""
    if not text:
        raise RuntimeError(f"{name} returned no visible content")
    return {
        "name": name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "wall_s": round(wall_s, 3),
        "usage": response.get("usage"),
        "timings": response.get("timings"),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "response": text,
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "reasoning": reasoning,
        "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
    }


def stream_completion(endpoint, model):
    prompt = "Write a compact 100-word explanation of quantized neural networks."
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 192,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        },
    )
    started = time.monotonic()
    first = None
    reasoning_first = None
    usage = None
    chunks = []
    reasoning_chunks = []
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            delta = choices[0].get("delta") or {} if choices else {}
            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")
            if reasoning:
                reasoning_first = reasoning_first or time.monotonic()
                reasoning_chunks.append(reasoning)
            if content:
                first = first or time.monotonic()
                chunks.append(content)
    ended = time.monotonic()
    text = "".join(chunks)
    reasoning = "".join(reasoning_chunks)
    if not text:
        raise RuntimeError("streaming_generation returned no visible content")
    return {
        "name": "streaming_generation",
        "prompt": prompt,
        "ttft_s": round(first - started, 3) if first else None,
        "reasoning_ttft_s": (
            round(reasoning_first - started, 3) if reasoning_first else None
        ),
        "wall_s": round(ended - started, 3),
        "chunks": len(chunks),
        "reasoning_chunks": len(reasoning_chunks),
        "usage": usage,
        "response": text,
        "reasoning": reasoning,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--gpu-mode", choices=("dual", "single"), required=True)
    parser.add_argument("--cold-start-seconds", type=float)
    parser.add_argument("--soak-seconds", type=int, default=300)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    run = {
        "schema_version": 2,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": args.endpoint,
        "model": args.model,
        "profile": args.profile,
        "gpu_mode": args.gpu_mode,
        "cold_start_seconds": args.cold_start_seconds,
        "cases": [],
    }
    health, health_s = request_json(args.endpoint + "/health", timeout=10)
    run["health"] = {"response": health, "wall_s": round(health_s, 3)}
    for case in CASES:
        result = completion(args.endpoint, args.model, *case)
        run["cases"].append(result)
        print(
            f"{result['name']}: wall={result['wall_s']}s "
            f"input={result.get('usage', {}).get('prompt_tokens')} "
            f"output={result.get('usage', {}).get('completion_tokens')} "
            f"input_tps={result.get('timings', {}).get('prompt_per_second')} "
            f"output_tps={result.get('timings', {}).get('predicted_per_second')}",
            flush=True,
        )
    run["stream"] = stream_completion(args.endpoint, args.model)

    probes = []
    soak_started = time.monotonic()
    while time.monotonic() - soak_started < args.soak_seconds:
        _, wall_s = request_json(args.endpoint + "/health", timeout=10)
        probes.append(
            {
                "elapsed_s": round(time.monotonic() - soak_started, 3),
                "wall_s": round(wall_s, 3),
            }
        )
        time.sleep(min(30, args.soak_seconds))
    run["soak"] = {
        "requested_s": args.soak_seconds,
        "actual_s": round(time.monotonic() - soak_started, 3),
        "health_probes": probes,
    }
    run["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output:
        output.write(json.dumps(run, separators=(",", ":")) + "\n")
    print(f"Recorded {args.output}", flush=True)


if __name__ == "__main__":
    main()
