#!/usr/bin/env python3
"""Verify both API formats through the private Portkey endpoint."""

import argparse
import collections
import json
import pathlib
import subprocess
import threading
import time
import urllib.request


def request(base_url: str, path: str, config: dict, payload: dict) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key",
            "x-portkey-config": json.dumps(config, separators=(",", ":")),
        },
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:14100")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--observe-backends", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).parent
    openai_config = json.loads((root / "openai-loadbalance.json").read_text())
    anthropic_config = json.loads((root / "anthropic-loadbalance.json").read_text())

    observed = collections.Counter()
    stop_sampling = threading.Event()

    def sample_connections() -> None:
        while not stop_sampling.is_set():
            sockets = subprocess.run(
                ["ss", "-tnH"], text=True, capture_output=True, check=True
            ).stdout
            for port in (19101, 19102):
                if f"172.30.0.1:{port}" in sockets:
                    observed[port] += 1
            time.sleep(0.01)

    sampler = None
    if args.observe_backends:
        sampler = threading.Thread(target=sample_connections, daemon=True)
        sampler.start()

    results = {"openai": [], "anthropic": []}
    for index in range(args.requests):
        openai = request(
            args.base_url,
            "/v1/chat/completions",
            openai_config,
            {
                "model": "gemma3n-e2b-q4",
                "messages": [{"role": "user", "content": f"Reply exactly: openai-{index}"}],
                "max_tokens": 16,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        results["openai"].append(openai["choices"][0]["message"]["content"].strip())

        anthropic = request(
            args.base_url,
            "/v1/messages",
            anthropic_config,
            {
                "model": "gemma3n-e2b-q4",
                "messages": [{"role": "user", "content": f"Reply exactly: anthropic-{index}"}],
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        results["anthropic"].append(
            "".join(
                block.get("text", "")
                for block in anthropic["content"]
                if block.get("type") == "text"
            ).strip()
        )

    stop_sampling.set()
    if sampler:
        sampler.join()
        if set(observed) != {19101, 19102}:
            raise RuntimeError(f"Did not observe both FRP backends: {dict(observed)}")

    if any(not text for texts in results.values() for text in texts):
        raise RuntimeError("Gateway returned an empty response")
    if observed:
        results["observed_backend_samples"] = dict(observed)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
