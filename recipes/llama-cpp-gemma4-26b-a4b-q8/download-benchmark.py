import json
import os
import pathlib
import subprocess
import sys
import time


HF_TOKEN = "__HF_TOKEN__"
HF_REPO = "unsloth/gemma-4-26B-A4B-it-GGUF"
MODEL_FILE = "gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"

started = time.monotonic()
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
dependencies_s = time.monotonic() - started

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HOME"] = "/tmp/huggingface"
from huggingface_hub import hf_hub_download

download_started = time.monotonic()
model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)
download_s = time.monotonic() - download_started
size = pathlib.Path(model_path).stat().st_size
meminfo = {
    key: int(value.split()[0]) * 1024
    for key, value in (
        line.split(":", 1)
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines()
    )
}
print(
    "HF_DOWNLOAD_BENCHMARK "
    + json.dumps(
        {
            "authenticated": True,
            "transport": "hf_xet_high_performance",
            "dependencies_s": round(dependencies_s, 3),
            "download_s": round(download_s, 3),
            "bytes": size,
            "mib_per_s": round(size / 1024 / 1024 / download_s, 3),
            "ram_total_gib": round(meminfo["MemTotal"] / 1024**3, 3),
            "ram_available_gib": round(meminfo["MemAvailable"] / 1024**3, 3),
        }
    ),
    flush=True,
)
