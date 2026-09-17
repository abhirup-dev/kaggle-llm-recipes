#!/usr/bin/env python3
"""Minimal Kaggle E2B llama.cpp server exposed through the generic FRP tunnel."""

import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

from frp_tunnel import TunnelConfig, run as run_tunnel
from observability import (
    LLAMA_LOG_PATH,
    configure_logger,
    emit,
    install_observability,
    start_fluent_bit,
    start_resource_sampler,
)

LLAMA_CPP_TAG = "b10107"
LLAMA_CPP_ARCHIVE = f"llama.cpp-{LLAMA_CPP_TAG}-cuda-12.8-amd64.tar.gz"
LLAMA_CPP_URL = (
    f"https://github.com/ai-dock/llama.cpp-cuda/releases/download/"
    f"{LLAMA_CPP_TAG}/{LLAMA_CPP_ARCHIVE}"
)
LLAMA_CPP_SHA256 = "7093813fc1f5d31d07d702b6c362c1bd96fab1de2f4d692f066bc29d39f0eee6"
MODEL_REPO = "unsloth/gemma-3n-E2B-it-GGUF"
MODEL_FILE = "gemma-3n-E2B-it-Q4_K_M.gguf"
MODEL_ALIAS = "gemma3n-e2b-q4"
PORT = 8080


def post_json(path: str, payload: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_health(process: subprocess.Popen) -> None:
    for _ in range(900):
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1):
                return
        except Exception:
            time.sleep(1)
    raise TimeoutError("llama-server did not become healthy")


secrets = __RUNTIME_SECRETS__
os.environ.update(secrets)
tunnel_config = TunnelConfig.from_env()
verify_public_ingress(tunnel_config)
log_relay_enabled = os.environ["LOG_RELAY_ENABLED"] == "1"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[hf_xet]"],
    check=True,
)
install_observability(log_relay_enabled)
from huggingface_hub import hf_hub_download

logger = configure_logger()
emit(logger, "worker_starting", worker_id=os.environ["LOG_WORKER_ID"])
fluent_bit = fluent_bit_log = None
if log_relay_enabled:
    fluent_bit, fluent_bit_log = start_fluent_bit(os.environ["LOG_WORKER_ID"])
    emit(
        logger,
        "log_relay_started",
        relay_host=os.environ["LOG_RELAY_HOST"],
        relay_port=int(os.environ["LOG_RELAY_PORT"]),
    )
else:
    emit(logger, "log_relay_disabled")

archive_path = pathlib.Path("/tmp") / LLAMA_CPP_ARCHIVE
urllib.request.urlretrieve(LLAMA_CPP_URL, archive_path)
import hashlib

if hashlib.sha256(archive_path.read_bytes()).hexdigest() != LLAMA_CPP_SHA256:
    raise RuntimeError("llama.cpp checksum mismatch")
subprocess.run(["tar", "-xzf", archive_path, "-C", "/tmp"], check=True)
server_binary = pathlib.Path("/tmp/cuda-12.8/llama-server")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    filter(None, [str(server_binary.parent), os.environ.get("LD_LIBRARY_PATH")])
)
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    token=os.environ["HF_TOKEN"],
    cache_dir="/tmp/huggingface",
)

server_log = LLAMA_LOG_PATH.open("a", buffering=1)
server = subprocess.Popen(
    [
        server_binary,
        "--model",
        model_path,
        "--alias",
        MODEL_ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "999",
        "--parallel",
        "1",
        "--jinja",
        "--reasoning",
        "off",
        "--metrics",
    ],
    stdout=server_log,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

wait_for_health(server)
resource_sampler_stop = start_resource_sampler(logger, server.pid)

openai_response = post_json(
    "/v1/chat/completions",
    {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
assert openai_response["choices"][0]["message"]["content"].strip()

anthropic_response = post_json(
    "/v1/messages",
    {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "max_tokens": 16,
        "temperature": 0,
    },
)
assert any(block.get("text", "").strip() for block in anthropic_response["content"])

emit(
    logger,
    "local_apis_ready",
    model=MODEL_ALIAS,
    openai="/v1/chat/completions",
    anthropic="/v1/messages",
    token_count="/v1/messages/count_tokens",
)

try:
    raise SystemExit(run_tunnel(tunnel_config))
finally:
    resource_sampler_stop.set()
    server.terminate()
    server_log.close()
    if fluent_bit:
        fluent_bit.terminate()
    if fluent_bit_log:
        fluent_bit_log.close()
