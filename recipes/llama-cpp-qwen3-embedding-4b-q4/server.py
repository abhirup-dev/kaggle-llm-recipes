import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import urllib.request


LLAMA_CPP_TAG = "b10107"
LLAMA_CPP_ARCHIVE = f"llama.cpp-{LLAMA_CPP_TAG}-cuda-12.8-amd64.tar.gz"
LLAMA_CPP_URL = (
    "https://github.com/ai-dock/llama.cpp-cuda/releases/download/"
    f"{LLAMA_CPP_TAG}/{LLAMA_CPP_ARCHIVE}"
)
LLAMA_CPP_SHA256 = "7093813fc1f5d31d07d702b6c362c1bd96fab1de2f4d692f066bc29d39f0eee6"
MODEL_REPO = "Qwen/Qwen3-Embedding-4B-GGUF"
MODEL_FILE = "Qwen3-Embedding-4B-Q4_K_M.gguf"
MODEL_ALIAS = "qwen3-embedding-4b-q4"
HF_TOKEN = "__HF_TOKEN__"
CTX_SIZE = int("__CTX_SIZE__")
BATCH_SIZE = int("__BATCH_SIZE__")
UBATCH_SIZE = int("__UBATCH_SIZE__")
PARALLEL = int("__PARALLEL__")
PORT = 8080
KEEPALIVE_INTERVAL_SECONDS = 10 * 60

started = time.monotonic()
stages = {}


def mark(name, **details):
    stages[name] = round(time.monotonic() - started, 3)
    print(json.dumps({"stage": name, "elapsed_s": stages[name], **details}), flush=True)


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


mark("dependencies_ready")

archive_path = pathlib.Path("/tmp") / LLAMA_CPP_ARCHIVE
install_dir = pathlib.Path("/tmp/llama.cpp")
urllib.request.urlretrieve(LLAMA_CPP_URL, archive_path)
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != LLAMA_CPP_SHA256:
    raise RuntimeError("llama.cpp archive checksum mismatch")
subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(install_dir.parent)], check=True)
server_binary = install_dir.parent / "cuda-12.8" / "llama-server"
if not server_binary.is_file():
    raise FileNotFoundError(f"Missing llama-server binary: {server_binary}")
mark("llama_cpp_ready", tag=LLAMA_CPP_TAG, source="ai-dock/llama.cpp-cuda")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    filter(None, [str(server_binary.parent), os.environ.get("LD_LIBRARY_PATH")])
)

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[hf_xet]"],
    check=True,
)
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    cache_dir="/tmp/huggingface",
)
mark(
    "model_ready",
    bytes=pathlib.Path(model_path).stat().st_size,
    source="hf_xet_high_performance",
)

log = open("/kaggle/working/llama-server.log", "w")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
server = subprocess.Popen(
    [
        str(server_binary),
        "--model",
        model_path,
        "--alias",
        MODEL_ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--ctx-size",
        str(CTX_SIZE),
        "--batch-size",
        str(BATCH_SIZE),
        "--ubatch-size",
        str(UBATCH_SIZE),
        "--n-gpu-layers",
        "999",
        "--embedding",
        "--pooling",
        "last",
        "--parallel",
        str(PARALLEL),
        "--cont-batching",
        "--flash-attn",
        "on",
        "--metrics",
    ],
    stdout=log,
    stderr=subprocess.STDOUT,
)

for _ in range(900):
    if server.poll() is not None:
        log.flush()
        raise RuntimeError(
            "llama-server exited during startup: "
            + pathlib.Path(log.name).read_text()[-12000:]
        )
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
        break
    except Exception:
        time.sleep(1)
else:
    raise TimeoutError("llama-server did not start within 15 minutes")
mark("llama_server_listening")

warm = post_json(
    f"http://127.0.0.1:{PORT}/v1/embeddings",
    {
        "model": MODEL_ALIAS,
        "input": "Instruct: Retrieve relevant passages\\nQuery: What is the capital of China?",
        "encoding_format": "float",
    },
    timeout=300,
)
warm_vector = warm["data"][0]["embedding"]
warm_norm = math.sqrt(sum(value * value for value in warm_vector))
if len(warm_vector) != 2560 or not 0.99 <= warm_norm <= 1.01:
    raise RuntimeError(
        f"Unexpected warm-up embedding: dimensions={len(warm_vector)}, norm={warm_norm}"
    )
mark("model_warmed", dimensions=len(warm_vector), l2_norm=round(warm_norm, 6))

mark("tunnel_ready", transport="local")


def resource_state():
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    return {"gpus": gpu}


ready = {
    "status": "ready",
    "model": MODEL_ALIAS,
    "model_repo": MODEL_REPO,
    "model_file": MODEL_FILE,
    "source": "hf_xet_high_performance",
    "url": f"http://127.0.0.1:{PORT}",
    "transport": "local",
    "ctx_size": CTX_SIZE,
    "batch_size": BATCH_SIZE,
    "ubatch_size": UBATCH_SIZE,
    "parallel": PARALLEL,
    "pooling": "last",
    "embedding_dimensions": len(warm_vector),
    "stages": stages,
    **resource_state(),
}
with open("/kaggle/working/endpoint.json", "w") as endpoint_file:
    json.dump(ready, endpoint_file, indent=2)
print("LLAMA_CPP_SERVER_READY " + json.dumps(ready), flush=True)

try:
    next_keepalive = time.monotonic() + KEEPALIVE_INTERVAL_SECONDS
    while True:
        if server.poll() is not None:
            log.flush()
            raise RuntimeError(
                f"llama-server exited with {server.returncode}: "
                + pathlib.Path(log.name).read_text()[-12000:]
            )
        if time.monotonic() < next_keepalive:
            time.sleep(1)
            continue
        next_keepalive = time.monotonic() + KEEPALIVE_INTERVAL_SECONDS
        try:
            health = get_json(f"http://127.0.0.1:{PORT}/health", timeout=5)
            print(
                json.dumps(
                    {
                        "stage": "keepalive_completed",
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "health": health,
                        **resource_state(),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "stage": "keepalive_failed",
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "error": repr(error),
                    }
                ),
                flush=True,
            )
finally:
    server.terminate()
    log.close()
