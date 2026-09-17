#!/usr/bin/env python3
"""Verify OpenAI and Anthropic endpoints with Portkey's Python client."""

import argparse
import json

from portkey_ai import Portkey


def client(base_url: str, protocol: str, ports: list[int], bridge_host: str) -> Portkey:
    return Portkey(
        api_key="no-key",
        base_url=base_url.rstrip("/") + "/v1",
        config={
            "strategy": {"mode": "loadbalance"},
            "targets": [
                {
                    "provider": protocol,
                    "api_key": "no-key",
                    "custom_host": f"http://{bridge_host}:{port}/v1",
                    "weight": 1,
                }
                for port in ports
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:14100")
    parser.add_argument("--api", choices=("openai", "anthropic", "both"), default="both")
    parser.add_argument("--model", default="gemma4-26b-a4b-ud-q8")
    parser.add_argument("--port", action="append", type=int, dest="ports")
    parser.add_argument("--bridge-host", default="172.30.0.1")
    args = parser.parse_args()
    ports = args.ports or [19101]
    results = {}

    if args.api in ("openai", "both"):
        openai = client(args.base_url, "openai", ports, args.bridge_host)
        response = openai.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply exactly: portkey-openai-ok"}],
            max_tokens=16,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        results["openai"] = response.choices[0].message.content

    if args.api in ("anthropic", "both"):
        claude = client(args.base_url, "anthropic", ports, args.bridge_host)
        response = claude.post(
            "/messages",
            model=args.model,
            messages=[{"role": "user", "content": "Reply exactly: portkey-anthropic-ok"}],
            max_tokens=16,
            temperature=0,
        )
        results["anthropic"] = "".join(
            block["text"] for block in response.content if block["type"] == "text"
        )

    if any(not value for value in results.values()):
        raise RuntimeError("SDK returned an empty response")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
