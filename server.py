import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request


MODEL = "gemma3n:e4b"
PORT = 11434
DOMAIN = "neurosis-washroom-gliding.ngrok-free.dev"
NGROK_AUTHTOKEN = "__NGROK_AUTHTOKEN__"
MAX_RUNTIME_SECONDS = 2 * 60 * 60

started = time.monotonic()
stages = {}


def mark(name):
    stages[name] = round(time.monotonic() - started, 3)
    print(json.dumps({"stage": name, "elapsed_s": stages[name]}), flush=True)


if not shutil.which("zstd"):
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)

if not shutil.which("ollama"):
    installer = "/tmp/install-ollama.sh"
    urllib.request.urlretrieve("https://ollama.com/install.sh", installer)
    subprocess.run(["sh", installer], check=True)
mark("ollama_installed")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ollama", "pyngrok"],
    check=True,
)
import ollama
from pyngrok import ngrok

mark("python_dependencies_ready")

log = open("/kaggle/working/ollama.log", "w")
server = subprocess.Popen(
    ["ollama", "serve"],
    env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{PORT}"},
    stdout=log,
    stderr=subprocess.STDOUT,
)

for _ in range(120):
    if server.poll() is not None:
        raise RuntimeError("Ollama exited during startup")
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/tags", timeout=1)
        break
    except Exception:
        time.sleep(1)
else:
    raise TimeoutError("Ollama did not start within 120 seconds")
mark("ollama_listening")

ngrok.set_auth_token(NGROK_AUTHTOKEN)
tunnel = ngrok.connect(
    addr=PORT,
    domain=DOMAIN,
    host_header=f"localhost:{PORT}",
)
mark("tunnel_ready")

client = ollama.Client(host=f"http://127.0.0.1:{PORT}")
client.pull(model=MODEL)
mark("model_downloaded")
client.generate(
    model=MODEL,
    prompt="Reply with exactly: ready",
    stream=False,
    options={"temperature": 0, "num_predict": 4},
    keep_alive="10m",
)
mark("model_warmed")

ready = {
    "status": "ready",
    "model": MODEL,
    "url": tunnel.public_url,
    "stages": stages,
}
with open("/kaggle/working/endpoint.json", "w") as endpoint_file:
    json.dump(ready, endpoint_file, indent=2)
print("OLLAMA_SERVER_READY " + json.dumps(ready), flush=True)

try:
    deadline = time.monotonic() + MAX_RUNTIME_SECONDS
    while server.poll() is None and time.monotonic() < deadline:
        time.sleep(30)
finally:
    ngrok.disconnect(tunnel.public_url)
    server.terminate()
    log.close()
