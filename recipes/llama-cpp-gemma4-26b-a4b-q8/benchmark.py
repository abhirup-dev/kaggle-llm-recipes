import json
import time
import urllib.request


ENDPOINT = "https://neurosis-washroom-gliding.ngrok-free.dev"
MODEL = "gemma4-26b-a4b-ud-q8"
CASES = [
    ("short", "Explain memory mapping in one sentence.", 128),
    (
        "long_prompt",
        ("Kaggle notebooks provide ephemeral compute and durable attached datasets. " * 120)
        + "Summarize the tradeoff in three bullets.",
        192,
    ),
    (
        "generation",
        "Write a technically precise 300-word explanation of mixture-of-experts models.",
        384,
    ),
]


def post(payload):
    request = urllib.request.Request(
        ENDPOINT + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        },
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    return result, time.monotonic() - started


for name, prompt, max_tokens in CASES:
    response, wall_s = post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": max_tokens,
        }
    )
    print(
        "INFERENCE_BENCHMARK "
        + json.dumps(
            {
                "case": name,
                "wall_s": round(wall_s, 3),
                "usage": response.get("usage"),
                "timings": response.get("timings"),
                "preview": response["choices"][0]["message"]["content"][:120],
            }
        ),
        flush=True,
    )

request = urllib.request.Request(
    ENDPOINT + "/metrics",
    headers={"ngrok-skip-browser-warning": "true"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    metrics = response.read().decode()
print(
    "\n".join(
        line
        for line in metrics.splitlines()
        if "tokens_" in line or "prompt_" in line or "predicted_" in line
    )
)
