#!/usr/bin/env python3
"""Verify OpenAI and Anthropic llama.cpp APIs through one base URL."""

import argparse
import json
import urllib.request


def post(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def stream_text(base_url: str, path: str, payload: dict, anthropic: bool) -> str:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps({**payload, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if anthropic:
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunks.append(delta.get("text", ""))
            else:
                choices = event.get("choices", [])
                if choices:
                    chunks.append(choices[0].get("delta", {}).get("content") or "")
    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError(f"{path} streaming response was empty")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--model", default="gemma3n-e2b-q4")
    args = parser.parse_args()

    openai = post(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: openai-ok"}],
            "max_tokens": 16,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    openai_text = openai["choices"][0]["message"]["content"].strip()
    if not openai_text:
        raise RuntimeError("OpenAI response was empty")

    anthropic = post(
        args.base_url,
        "/v1/messages",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: anthropic-ok"}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    anthropic_text = "".join(
        block.get("text", "") for block in anthropic["content"] if block.get("type") == "text"
    ).strip()
    if not anthropic_text:
        raise RuntimeError("Anthropic response was empty")

    token_count = post(
        args.base_url,
        "/v1/messages/count_tokens",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Count this request."}],
        },
    )
    if token_count["input_tokens"] <= 0:
        raise RuntimeError("Anthropic token count was not positive")

    openai_stream = stream_text(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: openai-stream-ok"}],
            "max_tokens": 20,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        anthropic=False,
    )
    anthropic_stream = stream_text(
        args.base_url,
        "/v1/messages",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: anthropic-stream-ok"}],
            "max_tokens": 20,
            "temperature": 0,
        },
        anthropic=True,
    )

    print(
        json.dumps(
            {
                "openai": openai_text,
                "openai_stream": openai_stream,
                "anthropic": anthropic_text,
                "anthropic_stream": anthropic_stream,
                "anthropic_input_tokens": token_count["input_tokens"],
            }
        )
    )


if __name__ == "__main__":
    main()
