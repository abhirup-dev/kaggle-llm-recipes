import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request


LLAMA_CPP_TAG = "b10107"
LLAMA_CPP_ARCHIVE = f"llama.cpp-{LLAMA_CPP_TAG}-cuda-12.8-amd64.tar.gz"
LLAMA_CPP_URL = (
    f"https://github.com/ai-dock/llama.cpp-cuda/releases/download/"
    f"{LLAMA_CPP_TAG}/{LLAMA_CPP_ARCHIVE}"
)
LLAMA_CPP_SHA256 = "7093813fc1f5d31d07d702b6c362c1bd96fab1de2f4d692f066bc29d39f0eee6"
MODEL_FILE = "gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
MODEL_ALIAS = "gemma4-26b-a4b-ud-q8"
MODEL_SOURCE_MODE = "__MODEL_SOURCE_MODE__"
PORT = 8080
DOMAIN = "neurosis-washroom-gliding.ngrok-free.dev"
NGROK_AUTHTOKEN = "__NGROK_AUTHTOKEN__"
HF_TOKEN = "__HF_TOKEN__"
MAX_RUNTIME_SECONDS = 2 * 60 * 60

started = time.monotonic()
stages = {}


def mark(name, **details):
    stages[name] = round(time.monotonic() - started, 3)
    print(
        json.dumps({"stage": name, "elapsed_s": stages[name], **details}),
        flush=True,
    )


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)
from pyngrok import ngrok

mark("dependencies_ready")

archive_path = pathlib.Path("/tmp") / LLAMA_CPP_ARCHIVE
install_dir = pathlib.Path("/tmp/llama.cpp")
urllib.request.urlretrieve(LLAMA_CPP_URL, archive_path)
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != LLAMA_CPP_SHA256:
    raise RuntimeError("llama.cpp archive checksum mismatch")
subprocess.run(
    ["tar", "-xzf", str(archive_path), "-C", str(install_dir.parent)],
    check=True,
)
server_binary = install_dir.parent / "cuda-12.8" / "llama-server"
if not server_binary.is_file():
    raise FileNotFoundError(f"Missing llama-server binary: {server_binary}")
mark("llama_cpp_ready", tag=LLAMA_CPP_TAG, source="ai-dock/llama.cpp-cuda")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    filter(None, [str(server_binary.parent), os.environ.get("LD_LIBRARY_PATH")])
)

dataset_models = list(pathlib.Path("/kaggle/input").glob(f"**/{MODEL_FILE}"))
if MODEL_SOURCE_MODE == "dataset":
    if len(dataset_models) != 1:
        raise RuntimeError(
            f"Expected exactly one attached {MODEL_FILE}, found {dataset_models}"
        )
    model_path = str(dataset_models[0])
    model_source = "kaggle_dataset_mmap"
else:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "huggingface_hub[hf_xet]",
        ],
        check=True,
    )
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename=MODEL_FILE,
        cache_dir="/tmp/huggingface",
    )
    model_source = "hf_xet_high_performance"
mark("model_ready", bytes=pathlib.Path(model_path).stat().st_size, source=model_source)

log = open("/kaggle/working/llama-server.log", "w")
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
        "4096",
        "--n-gpu-layers",
        "999",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--parallel",
        "1",
        "--fit",
        "on",
        "--fit-target",
        "768",
        "--jinja",
        "--metrics",
        "--temp",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "64",
    ],
    stdout=log,
    stderr=subprocess.STDOUT,
)

for _ in range(900):
    if server.poll() is not None:
        raise RuntimeError("llama-server exited during startup")
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
        break
    except Exception:
        time.sleep(1)
else:
    raise TimeoutError("llama-server did not start within 15 minutes")
mark("llama_server_listening")

warm = post_json(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    {
        "model": MODEL_ALIAS,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: ready",
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 96,
    },
    timeout=300,
)
warm_text = warm["choices"][0]["message"]["content"].strip()
if not warm_text:
    raise RuntimeError("Warm-up returned no visible content")
mark("model_warmed", response=warm_text)

ngrok.set_auth_token(NGROK_AUTHTOKEN)
tunnel = ngrok.connect(
    addr=PORT,
    domain=DOMAIN,
    host_header=f"localhost:{PORT}",
)
mark("tunnel_ready")

gpu_state = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().splitlines()
ready = {
    "status": "ready",
    "model": MODEL_ALIAS,
    "source": model_source,
    "model_path": model_path,
    "url": tunnel.public_url,
    "stages": stages,
    "gpus": gpu_state,
}
with open("/kaggle/working/endpoint.json", "w") as endpoint_file:
    json.dump(ready, endpoint_file, indent=2)
print("LLAMA_CPP_SERVER_READY " + json.dumps(ready), flush=True)

try:
    deadline = time.monotonic() + MAX_RUNTIME_SECONDS
    while server.poll() is None and time.monotonic() < deadline:
        time.sleep(30)
finally:
    ngrok.disconnect(tunnel.public_url)
    server.terminate()
    log.close()
