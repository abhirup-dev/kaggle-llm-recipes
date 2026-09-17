#!/usr/bin/env python3
"""Gemma 3n E2B multimodal OpenAI-compatible server exposed through FRP."""

import base64
import io
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from frp_tunnel import TunnelConfig, run as run_tunnel
from observability import (
    configure_logger,
    emit,
    install_observability,
    start_fluent_bit,
    start_resource_sampler,
)

MODEL_REPO = "google/gemma-3n-E2B-it"
MODEL_ALIAS = "gemma3n-e2b-multimodal"
PORT = 8080
MAX_REQUEST_BYTES = 16 * 1024 * 1024

secrets = __RUNTIME_SECRETS__
os.environ.update(secrets)
tunnel_config = TunnelConfig.from_env()
verify_public_ingress(tunnel_config)
log_relay_enabled = os.environ["LOG_RELAY_ENABLED"] == "1"

import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "accelerate",
        "pillow",
        "timm",
        "transformers",
    ],
    check=True,
)
install_observability(log_relay_enabled)

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = configure_logger()
emit(logger, "worker_starting", worker_id=os.environ["LOG_WORKER_ID"])
fluent_bit = fluent_bit_log = None
if log_relay_enabled:
    fluent_bit, fluent_bit_log = start_fluent_bit(os.environ["LOG_WORKER_ID"])
processor = AutoProcessor.from_pretrained(MODEL_REPO, token=os.environ["HF_TOKEN"])
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_REPO,
    device_map="auto",
    token=os.environ["HF_TOKEN"],
    torch_dtype=torch.bfloat16,
)
model.eval()
resource_sampler_stop = start_resource_sampler(logger, os.getpid())
emit(logger, "model_ready", model=MODEL_ALIAS)


def image_from_url(url: str) -> Image.Image:
    if url.startswith("data:image/"):
        try:
            payload = url.split(",", 1)[1]
            raw = base64.b64decode(payload, validate=True)
        except (IndexError, ValueError) as error:
            raise ValueError("invalid image data URL") from error
    elif url.startswith(("https://", "http://")):
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read(MAX_REQUEST_BYTES + 1)
    else:
        raise ValueError("image URL must be HTTP(S) or a data URL")
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("image is too large")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def complete(payload: dict) -> dict:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    normalized = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image_url":
                    parts.append(
                        {
                            "type": "image",
                            "image": image_from_url(part["image_url"]["url"]),
                        }
                    )
                elif part.get("type") == "text":
                    parts.append({"type": "text", "text": part["text"]})
            content = parts
        elif isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({"role": message["role"], "content": content})

    inputs = processor.apply_chat_template(
        normalized,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs = {
        name: value.to(model.dtype) if value.is_floating_point() else value
        for name, value in inputs.items()
    }
    max_tokens = min(int(payload.get("max_tokens", 128)), 512)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    generated = output[0]
    input_length = inputs["input_ids"].shape[-1]
    if generated.shape[-1] > input_length:
        generated = generated[input_length:]
    text = processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()
    if not text:
        emit(
            logger,
            "empty_generation",
            input_length=input_length,
            output_length=int(output[0].shape[-1]),
            token_ids=output[0][-32:].tolist(),
            decoded=processor.decode(output[0][-32:], skip_special_tokens=False),
        )
    return {
        "id": "chatcmpl-kaggle",
        "object": "chat.completion",
        "model": MODEL_ALIAS,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.send_json(200, {"status": "ok"} if self.path == "/health" else {})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            self.send_json(200, complete(json.loads(self.rfile.read(length))))
        except Exception as error:
            emit(logger, "request_failed", error=str(error))
            self.send_json(400, {"error": {"message": str(error), "type": "invalid_request_error"}})

    def log_message(self, _format: str, *_args) -> None:
        return


server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
for _ in range(60):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1):
            break
    except Exception:
        time.sleep(1)
else:
    raise TimeoutError("multimodal server did not become healthy")

smoke_image = io.BytesIO()
Image.new("RGB", (32, 32), "red").save(smoke_image, format="PNG")
smoke_request = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=json.dumps(
        {
            "model": MODEL_ALIAS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Name the color in this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(smoke_image.getvalue()).decode()
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 16,
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(smoke_request, timeout=300) as response:
    smoke_response = json.load(response)
assert smoke_response["choices"][0]["message"]["content"].strip()

emit(logger, "local_api_ready", model=MODEL_ALIAS, openai="/v1/chat/completions")
try:
    raise SystemExit(run_tunnel(tunnel_config))
finally:
    resource_sampler_stop.set()
    server.shutdown()
    if fluent_bit:
        fluent_bit.terminate()
    if fluent_bit_log:
        fluent_bit_log.close()
