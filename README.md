# Kaggle LLM Recipes

Host Gemma 4 on a private Kaggle GPU kernel and query it from a local machine.

## Recommended recipe

The crystallized setup is the llama.cpp recipe for Unsloth Gemma 4 26B A4B
Dynamic Q8. It uses a private Kaggle Dataset for the fixed GGUF, exposes an
OpenAI-compatible endpoint through ngrok, and includes the measured cold-start
and token-throughput results:

[`recipes/llama-cpp-gemma4-26b-a4b-q8/README.md`](recipes/llama-cpp-gemma4-26b-a4b-q8/README.md).

Start it with:

```sh
./recipes/llama-cpp-gemma4-26b-a4b-q8/start-server
```

## Legacy Ollama recipe

The original Ollama setup remains below as the experimental record. Its Kaggle
kernels and cache datasets were deleted after the llama.cpp dataset-backed
recipe became the selected approach.

### Start the server

Prerequisites:

- Kaggle CLI installed and authenticated as `devabhirupdas`
- Kaggle account approved for GPU and internet access
- The fixed ngrok domain and `.secrets/ngrok.env` token remain valid

Run:

```sh
./start-server
```

The default is the largest Gemma 4 quantization expected to remain fully
GPU-resident across Kaggle's two 16 GB T4s:

```sh
./start-server                         # gemma4:31b-it-qat, 19 GB
./start-server gemma4:26b-a4b-it-qat  # smaller/faster MoE alternative
```

The command uploads a private Kaggle kernel, requests a T4 GPU, and waits up to
15 minutes for:

```text
Ready: https://neurosis-washroom-gliding.ngrok-free.dev
```

The kernel shuts itself down after two hours so an abandoned run does not keep
using GPU quota. Change `MAX_RUNTIME_SECONDS` in `server.py` when a different
bound is needed.

### Query from the Mac

```sh
curl -sS https://neurosis-washroom-gliding.ngrok-free.dev/api/generate \
  -H 'Content-Type: application/json' \
  -H 'ngrok-skip-browser-warning: true' \
  -d '{
    "model": "gemma4:31b-it-qat",
    "prompt": "Reply with exactly: hello from Kaggle",
    "stream": false
  }'
```

Check whether the server is live:

```sh
curl -fsS \
  -H 'ngrok-skip-browser-warning: true' \
  https://neurosis-washroom-gliding.ngrok-free.dev/api/tags
```

Inspect the batch run:

```sh
kaggle kernels status devabhirupdas/ollama-gemma-4-remote-server
kaggle kernels output devabhirupdas/ollama-gemma-4-remote-server -p outputs
```

Kernel page:

<https://www.kaggle.com/code/devabhirupdas/ollama-gemma-4-remote-server>

### Cold-start comparison on Kaggle T4 x2

Both 2026-07-25 runs used `gemma3n:e2b`, a fresh private batch kernel, and
stopped only after a real generation succeeded.

| Model source | Push to first generation | In-kernel ready | Warm request |
|---|---:|---:|---:|
| Direct `ollama pull` | **212.7s** | 206.9s | 1.83s |
| Attached cache + mmap | 361.3s | **188.4s** | 1.82s |

The cache saved 18.5 seconds inside Python, but Kaggle spent about 173 seconds
before the cached script began versus about 6 seconds for the pull run.
Attaching 5.24 GB therefore made end-to-end startup 148.6 seconds slower in
this comparison. Warm throughput was similar: about 240 input tok/s and
41–46 output tok/s.

An earlier cached `gemma3n:e4b` run became ready inside Python in 197.0
seconds. The smaller `e2b` cache only saved 8.6 seconds because cold model
loading and first generation dominate after the files are available.

Ollama reports durations in nanoseconds:

```text
prompt tokens/s = prompt_eval_count / prompt_eval_duration * 1e9
output tokens/s = eval_count / eval_duration * 1e9
```

### Security

This is deliberately the simplest unattended setup: `start-server` renders the
ngrok token into the temporary private Kaggle kernel source because Kaggle CLI
batch runs returned HTTP 400 when reading the same token through
`UserSecretsClient`.

The durable token is stored in the Git-ignored `.secrets/ngrok.env`. The public
repository contains only a placeholder. Keep the generated Kaggle kernel
private and rotate the token if that source becomes public. The exposed Ollama
endpoint currently has no application-level authentication.

### Why the Ollama runtime has no datasets

In the measured `gemma3n:e2b` comparison,
direct `ollama pull` reached a real generation in 212.7 seconds versus 361.3
seconds for an attached 5.24 GB cache. The e2b/e4b caches, cache smoke test,
and GitHub-source snapshot were deleted after that result. GitHub is the
canonical source and `kaggle kernels push` uploads the runtime directly.

### Gemma 4 capacity

Kaggle provides two 16 GB T4 GPUs. Ollama automatically spreads a model across
available GPUs when it cannot fit on one. `gemma4:31b-it-qat` is the largest
parameter variant and its 19 GB artifact leaves enough aggregate VRAM for a
small context and runtime overhead. The server fixes:

- context length: 4,096
- KV cache: q8_0
- parallel requests: 1
- flash attention: enabled

Verified on 2026-07-25:

| Measurement | Result |
|---|---:|
| Push to successful cold generation | 340.73s |
| Ollama installed | 34.899s |
| Model pull complete | 187.741s |
| Model warm complete | 335.087s |
| Model artifact / VRAM | 18,938,006,076 bytes |
| Warm request | 2.90s |
| Warm prompt processing | 62.0 tok/s |
| Warm generation | 12.65 tok/s |

Ollama reported `size_vram == size`, confirming the 30.7B Q4_0 model was
fully GPU-resident rather than partially offloaded to CPU. The 20 GB
`gemma4:31b-it-q4_K_M` should also fit, but the tested 19 GB QAT build leaves
more safety margin. The 34 GB Q8 and 63 GB BF16 variants exceed the 32 GB
aggregate VRAM before KV-cache/runtime overhead.

### Current limitations

- A fresh kernel installs Ollama, pulls the model, and loads it onto the GPUs.
- Cold-start time varies with Kaggle scheduling and input staging; one run is
  directional rather than a full statistical benchmark.
- Kaggle CLI launches a committed batch kernel, not an interactive draft.
- The documented CLI can inspect status/output but does not provide a reliable
  command for cancelling a live batch kernel.
- Kaggle and ngrok availability, quotas, and policies still apply.
