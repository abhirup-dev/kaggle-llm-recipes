# Gemma 4 26B A4B Dynamic Q8 with llama.cpp

This recipe is independent of the repository's Ollama recipe. It runs
Unsloth's Gemma 4 26B A4B Dynamic 8-bit GGUF with a pinned CUDA
`llama-server`, exposes an OpenAI-compatible API through ngrok, and records
llama.cpp's native token timings.

## Exact artifacts

- Model: `unsloth/gemma-4-26B-A4B-it-GGUF`
- File: `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf`
- Size: `27,636,232,928` bytes
- SHA-256: `d4bf9791d727d7b88aeea89aba309c68086a4d51cf337047c4e51dde7e243058`
- Private dataset: <https://www.kaggle.com/datasets/devabhirupdas/gemma-4-26b-a4b-ud-q8-gguf>
- Server kernel: <https://www.kaggle.com/code/devabhirupdas/llama-cpp-gemma-4-26b-a4b-dynamic-q8-server>
- Direct dataset uploader: <https://www.kaggle.com/code/devabhirupdas/gemma-4-q8-direct-dataset-uploader>
- Download benchmark: <https://www.kaggle.com/code/devabhirupdas/gemma-4-q8-hugging-face-download-benchmark>
- llama.cpp build: `ai-dock/llama.cpp-cuda` tag `b10107`

The runtime requests full GPU offload, a 4,096-token context, q8 KV cache,
one request slot, flash attention, and an even layer split across two T4s.

## Start and query

Dataset mode is the default and fails rather than silently downloading if the
attachment is absent:

```sh
./recipes/llama-cpp-gemma4-26b-a4b-q8/start-server
```

Run the same kernel without the dataset for an authenticated Xet comparison:

```sh
./recipes/llama-cpp-gemma4-26b-a4b-q8/start-server hf
```

Query the OpenAI-compatible endpoint:

```sh
curl -sS https://neurosis-washroom-gliding.ngrok-free.dev/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'ngrok-skip-browser-warning: true' \
  -d '{
    "model": "gemma4-26b-a4b-ud-q8",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 512
  }'
```

Gemma may return private deliberation in `message.reasoning_content` and the
answer in `message.content`. The verified factual request returned
`The capital of France is Paris.`

Run the repeatable benchmark:

```sh
./recipes/llama-cpp-gemma4-26b-a4b-q8/benchmark
```

## Measured results

All measurements below are from 2026-07-26. Dataset creation/upload time is
excluded from cold start.

### Kaggle host

| Resource | Observed |
|---|---:|
| CPU | 4 vCPUs |
| Host RAM | 31.348 GiB |
| Initially available RAM | 30.486 GiB |
| Writable disk | 19.518 GiB |
| Initially free writable disk | 19.502 GiB |
| GPUs | 2 × Tesla T4 |
| VRAM | 2 × 15,360 MiB |

The GGUF is larger than the writable disk. A notebook-output provisioner
therefore failed at 19 GiB with `ENOSPC`; it cannot create this dataset by
staging the complete file under `/kaggle/working`.

### One-time direct dataset upload

The CPU-only uploader streams Hugging Face directly into Kaggle's resumable
dataset API and uses a zero-allocation sparse placeholder. It does not stage
the GGUF on disk.

| Stage | Result |
|---|---:|
| Sustained transfer | 105.8–105.9 MiB/s |
| Upload complete | 252.20s |
| Dataset object created | 257.62s |
| Final file size | 27,636,232,928 bytes |

This is provisioning, not cold start, and is paid only when creating a model
dataset version.

Recreate or replace the private dataset with:

```sh
./recipes/llama-cpp-gemma4-26b-a4b-q8/upload-dataset-kaggle
```

### Dataset cold start

Kernel version 7 used the attached dataset and reached:

| Stage | Elapsed |
|---|---:|
| Python dependencies ready | 4.910s |
| Pinned llama.cpp ready | 7.286s |
| Dataset path verified | 7.304s |
| llama-server listening locally | 303.592s |
| First successful warm response | **358.868s** |
| ngrok client reported tunnel ready | 360.924s |

At readiness, `nvidia-smi` reported:

```text
Tesla T4, 15360 MiB total, 13629 MiB used
Tesla T4, 15360 MiB total, 13465 MiB used
```

The first successful request from the Mac was delayed until 1,745.74s because
the fixed ngrok domain continued returning 404 long after the notebook logged
`tunnel_ready`. Keep the model cold-start number (358.868s) and ngrok routing
delay separate.

### Dataset versus authenticated Hugging Face

| Source/run | Result |
|---|---:|
| Attached dataset, local warm response | **358.868s** |
| Authenticated Xet, exceptional CPU download | 118.523s at 222.369 MiB/s |
| Authenticated Xet, repeated CPU run | >19m; canceled |
| Authenticated Xet, same GPU server kernel | No `model_ready` after 497s; canceled at 529.8s |

The attached dataset is the fastest reliable course for this model on this
Kaggle shape. It removes a highly variable external download and produced a
fully warmed model before the comparison Xet run had even finished fetching
the file. The 118.523s Xet result is a useful best case, not a dependable cold
start.

### Inference

llama.cpp returns these per-request measurements in the top-level `timings`
object:

| Case | Input tokens | Output tokens | Wall | Prompt tok/s | Output tok/s |
|---|---:|---:|---:|---:|---:|
| Short | 23 | 128 | 3.893s | 62.79 | 42.29 |
| Long prompt | 1,345 | 192 | 9.981s | **742.25** | 25.41 |
| Generation-heavy | 35 | 384 | 9.853s | 145.76 | **43.99** |
| Factual visible-answer check | 28 | 115 | 4.1s client-side | 77.39 | 42.15 |

The `/metrics` endpoint also exposed cumulative Prometheus counters and
gauges, including `llamacpp:prompt_tokens_total`,
`llamacpp:tokens_predicted_total`, `llamacpp:prompt_tokens_seconds`, and
`llamacpp:predicted_tokens_seconds`.

## Security and lifecycle

- `.secrets/ngrok.env` and `.secrets/huggingface.env` are durable locally and
  ignored by Git.
- Tokens are rendered only into temporary private Kaggle kernel source.
- The ngrok endpoint has no application-level authentication.
- The server self-terminates after two hours.
- Both GPU benchmark sessions were manually stopped and verified as
  `CANCEL_ACKNOWLEDGED`.

Sources:

- <https://unsloth.ai/docs/models/gemma-4>
- <https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF>
- <https://github.com/ai-dock/llama.cpp-cuda>
- <https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md>
