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
MODEL_PROFILE = "__MODEL_PROFILE__"
if MODEL_PROFILE == "e2b":
    MODEL_REPO = "unsloth/gemma-3n-E2B-it-GGUF"
    MODEL_FILE = "gemma-3n-E2B-it-Q4_K_M.gguf"
    MODEL_ALIAS = "gemma3n-e2b-q4"
    TAILSCALE_HOSTNAME = "kaggle-gemma-e2b"
elif MODEL_PROFILE == "e4b":
    MODEL_REPO = "unsloth/gemma-3n-E4B-it-GGUF"
    MODEL_FILE = "gemma-3n-E4B-it-Q4_K_M.gguf"
    MODEL_ALIAS = "gemma3n-e4b-q4"
    TAILSCALE_HOSTNAME = "kaggle-gemma-e4b"
else:
    MODEL_REPO = "unsloth/gemma-4-26B-A4B-it-GGUF"
    MODEL_FILE = "gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
    MODEL_ALIAS = "gemma4-26b-a4b-ud-q8"
    TAILSCALE_HOSTNAME = "kaggle-gemma4"
MODEL_SOURCE_MODE = "__MODEL_SOURCE_MODE__"
TRANSPORT_MODE = "__TRANSPORT_MODE__"
GPU_MODE = "__GPU_MODE__"
CTX_SIZE = int("__CTX_SIZE__")
GPU_LAYERS = int("__GPU_LAYERS__")
KV_CACHE_TYPE = "__KV_CACHE_TYPE__"
N_CPU_MOE = int("__N_CPU_MOE__")
TENSOR_SPLIT = "__TENSOR_SPLIT__"
PORT = 8080
DOMAIN = "neurosis-washroom-gliding.ngrok-free.dev"
NGROK_AUTHTOKEN = "__NGROK_AUTHTOKEN__"
TAILSCALE_AUTH_KEY = "__TAILSCALE_AUTH_KEY__"
HF_TOKEN = "__HF_TOKEN__"
KEEPALIVE_INTERVAL_SECONDS = 10 * 60
TAILSCALE_VERSION = "1.98.9"
TAILSCALE_SHA256 = "11be30ad301d48f84ff52fec34f8a2f78eb3e3dee1be4e9624d19fccc8df5540"

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


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


if TRANSPORT_MODE == "ngrok":
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
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir="/tmp/huggingface",
    )
    model_source = "hf_xet_high_performance"
mark("model_ready", bytes=pathlib.Path(model_path).stat().st_size, source=model_source)

log = open("/kaggle/working/llama-server.log", "w")
if GPU_MODE == "single":
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
        "--n-gpu-layers",
        str(GPU_LAYERS),
        "--split-mode",
        "layer",
        "--tensor-split",
        "1" if GPU_MODE == "single" else TENSOR_SPLIT,
        "--n-cpu-moe",
        str(N_CPU_MOE),
        "--flash-attn",
        "on",
        "--cache-type-k",
        KV_CACHE_TYPE,
        "--cache-type-v",
        KV_CACHE_TYPE,
        "--parallel",
        "1",
        "--fit",
        "__FIT_MODE__",
        "--fit-target",
        "768",
        "--jinja",
        "--reasoning",
        "off",
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
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=300,
)
warm_text = warm["choices"][0]["message"]["content"].strip()
if not warm_text:
    raise RuntimeError("Warm-up returned no visible content")
mark("model_warmed", response=warm_text)

tailscaled = None
tailscale_log = None
tailscale_binary = None
tailscale_socket = None
if TRANSPORT_MODE == "ngrok":
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    tunnel = ngrok.connect(
        addr=PORT,
        bind_tls=True,
        host_header=f"localhost:{PORT}",
    )
    endpoint_url = tunnel.public_url
elif TRANSPORT_MODE in ("tailscale", "tailscale-join"):
    archive = pathlib.Path("/tmp/tailscale.tgz")
    urllib.request.urlretrieve(
        f"https://pkgs.tailscale.com/stable/"
        f"tailscale_{TAILSCALE_VERSION}_amd64.tgz",
        archive,
    )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != TAILSCALE_SHA256:
        raise RuntimeError("Tailscale archive checksum mismatch")
    subprocess.run(["tar", "-xzf", str(archive), "-C", "/tmp"], check=True)
    tailscale_dir = pathlib.Path(f"/tmp/tailscale_{TAILSCALE_VERSION}_amd64")
    tailscale_binary = tailscale_dir / "tailscale"
    tailscaled_binary = tailscale_dir / "tailscaled"
    tailscale_socket = "/tmp/tailscaled.sock"
    tailscale_log = open("/kaggle/working/tailscaled.log", "w")
    tailscaled = subprocess.Popen(
        [
            str(tailscaled_binary),
            "--tun=userspace-networking",
            "--state=mem:",
            f"--socket={tailscale_socket}",
        ],
        stdout=tailscale_log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        if pathlib.Path(tailscale_socket).exists():
            break
        if tailscaled.poll() is not None:
            raise RuntimeError("tailscaled exited during startup")
        time.sleep(0.5)
    else:
        raise TimeoutError("tailscaled socket did not appear")

    auth_key_file = pathlib.Path("/tmp/tailscale-auth-key")
    auth_key_file.write_text(TAILSCALE_AUTH_KEY)
    auth_key_file.chmod(0o600)
    try:
        up = subprocess.run(
            [
                str(tailscale_binary),
                f"--socket={tailscale_socket}",
                "up",
                f"--auth-key=file:{auth_key_file}",
                f"--hostname={TAILSCALE_HOSTNAME}",
                "--accept-dns=false",
            ],
            text=True,
            capture_output=True,
        )
    finally:
        auth_key_file.unlink(missing_ok=True)
    if up.returncode:
        raise RuntimeError(f"tailscale up failed: {up.stderr.strip()}")

    if TRANSPORT_MODE == "tailscale":
        subprocess.run(
            [
                str(tailscale_binary),
                f"--socket={tailscale_socket}",
                "serve",
                "--bg",
                f"--tcp={PORT}",
                f"tcp://127.0.0.1:{PORT}",
            ],
            check=True,
        )
    status = json.loads(
        subprocess.check_output(
            [
                str(tailscale_binary),
                f"--socket={tailscale_socket}",
                "status",
                "--json",
            ],
            text=True,
        )
    )
    endpoint_url = (
        f"http://{status['Self']['TailscaleIPs'][0]}:{PORT}"
        if TRANSPORT_MODE == "tailscale"
        else f"http://127.0.0.1:{PORT}"
    )
else:
    endpoint_url = f"http://127.0.0.1:{PORT}"
mark("tunnel_ready", transport=TRANSPORT_MODE)


def resource_state():
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    memory = {}
    for source, keys in (
        ("/proc/meminfo", {"MemTotal", "MemAvailable"}),
        (f"/proc/{server.pid}/status", {"VmRSS", "VmHWM"}),
    ):
        for line in pathlib.Path(source).read_text().splitlines():
            key, value = line.split(":", 1)
            if key in keys:
                memory[key] = value.strip()
    return {"gpus": gpu, "memory": memory}


resources = resource_state()
ready = {
    "status": "ready",
    "model": MODEL_ALIAS,
    "source": model_source,
    "model_path": model_path,
    "url": endpoint_url,
    "transport": TRANSPORT_MODE,
    "gpu_mode": GPU_MODE,
    "ctx_size": CTX_SIZE,
    "gpu_layers": GPU_LAYERS,
    "kv_cache_type": KV_CACHE_TYPE,
    "n_cpu_moe": N_CPU_MOE,
    "tensor_split": TENSOR_SPLIT,
    "stages": stages,
    **resources,
}
with open("/kaggle/working/endpoint.json", "w") as endpoint_file:
    json.dump(ready, endpoint_file, indent=2)
print("LLAMA_CPP_SERVER_READY " + json.dumps(ready), flush=True)

try:
    next_keepalive = time.monotonic()
    while True:
        if server.poll() is not None:
            log.flush()
            print(
                "LLAMA_CPP_SERVER_EXITED "
                + json.dumps(
                    {
                        "returncode": server.returncode,
                        "log_tail": pathlib.Path(log.name).read_text()[-12000:],
                    }
                ),
                flush=True,
            )
            raise RuntimeError(f"llama-server exited with {server.returncode}")
        if time.monotonic() < next_keepalive:
            time.sleep(1)
            continue
        next_keepalive = time.monotonic() + KEEPALIVE_INTERVAL_SECONDS
        try:
            slots = get_json(f"http://127.0.0.1:{PORT}/slots", timeout=2)
            if any(slot.get("is_processing") for slot in slots):
                print(
                    json.dumps(
                        {
                            "stage": "keepalive_skipped_busy",
                            "elapsed_s": round(time.monotonic() - started, 3),
                        }
                    ),
                    flush=True,
                )
                continue
            keepalive = post_json(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                {
                    "model": MODEL_ALIAS,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Are you alive? Reply with exactly: yes",
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 8,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=30,
            )
            answer = keepalive["choices"][0]["message"]["content"].strip()
            if not answer:
                raise RuntimeError("Keepalive returned no visible content")
            print(
                json.dumps(
                    {
                        "stage": "keepalive_completed",
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "answer": answer,
                        "timings": keepalive.get("timings"),
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
    if TRANSPORT_MODE == "ngrok":
        ngrok.disconnect(endpoint_url)
    elif TRANSPORT_MODE in ("tailscale", "tailscale-join") and tailscaled is not None:
        subprocess.run(
            [
                str(tailscale_binary),
                f"--socket={tailscale_socket}",
                "logout",
            ],
            capture_output=True,
        )
        tailscaled.terminate()
        tailscale_log.close()
    server.terminate()
    log.close()
