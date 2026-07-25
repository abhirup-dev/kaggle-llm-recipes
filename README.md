# Kaggle LLM Recipes

Run `gemma3n:e4b` with Ollama on a private Kaggle GPU kernel and expose its
Ollama API through a fixed ngrok domain.

## Start the server

Prerequisites:

- Kaggle CLI installed and authenticated as `devabhirupdas`
- Kaggle account approved for GPU and internet access
- The fixed ngrok domain and `.secrets/ngrok.env` token remain valid

Run:

```sh
./start-server
```

Verified on 2026-07-25: private kernel version 4 completed this flow without
browser interaction and returned `unattended CLI inference works`.

The kernel attaches two private datasets:

- `devabhirupdas/kaggle-llm-recipes-github-source`: a snapshot imported from
  this GitHub repository
- `devabhirupdas/gemma3n-e4b-ollama-cache`: the Ollama manifest and model blobs

The command uploads a private Kaggle kernel, requests a T4 GPU, and waits up to
15 minutes for:

```text
Ready: https://neurosis-washroom-gliding.ngrok-free.dev
```

The kernel shuts itself down after two hours so an abandoned run does not keep
using GPU quota. Change `MAX_RUNTIME_SECONDS` in `server.py` when a different
bound is needed.

## Query from the Mac

```sh
curl -sS https://neurosis-washroom-gliding.ngrok-free.dev/api/generate \
  -H 'Content-Type: application/json' \
  -H 'ngrok-skip-browser-warning: true' \
  -d '{
    "model": "gemma3n:e4b",
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
kaggle kernels status devabhirupdas/ollama-gemma-3n-remote-server
kaggle kernels output devabhirupdas/ollama-gemma-3n-remote-server -p outputs
```

Kernel page:

<https://www.kaggle.com/code/devabhirupdas/ollama-gemma-3n-remote-server>

## Performance observed on Kaggle T4 x2

| Scenario | Wall time | Input throughput | Output throughput |
|---|---:|---:|---:|
| Cold start through first answer | ~7m 11s | — | — |
| First cold request | 121.6s | 0.35 tok/s | 29.7 tok/s |
| Warm short text | 10.2s | 90.8 tok/s | 39.1 tok/s |
| Warm long input | 7.7s | 1,084.5 tok/s | 38.8 tok/s |
| Warm streaming | 5.1s; TTFT 1.54s | 127.7 tok/s | 39.6 tok/s |

Ollama reports durations in nanoseconds:

```text
prompt tokens/s = prompt_eval_count / prompt_eval_duration * 1e9
output tokens/s = eval_count / eval_duration * 1e9
```

## Security

This is deliberately the simplest unattended setup: `start-server` renders the
ngrok token into the temporary private Kaggle kernel source because Kaggle CLI
batch runs returned HTTP 400 when reading the same token through
`UserSecretsClient`.

The durable token is stored in the Git-ignored `.secrets/ngrok.env`. The public
repository contains only a placeholder. Keep the generated Kaggle kernel
private and rotate the token if that source becomes public. The exposed Ollama
endpoint currently has no application-level authentication.

## GitHub source and model cache datasets

Kaggle can create a dataset from a public GitHub repository archive. The
result is a versioned snapshot with the repository URL recorded as its remote
source; it is not a live Git checkout. Use the dataset page's **Update** action
after pushing repository changes:

<https://www.kaggle.com/datasets/devabhirupdas/kaggle-llm-recipes-github-source>

Kaggle datasets are durable and attach read-only under `/kaggle/input`. A
private dataset containing Ollama's manifest and referenced blobs can remove
the repeated 7.5 GB registry download. It does not remove the time needed to
load the model onto the GPU.

Prepare the cache from a machine that already has the model:

```sh
ollama pull gemma3n:e4b
./build-model-cache
```

Review and add the Gemma notice/terms before uploading:

```sh
kaggle datasets create \
  -p dataset-gemma3n-e4b \
  --keep-tabular
```

Attach it by adding the dataset slug to `dataset_sources` in
`kernel-metadata.json`. The server constructs a writable `OLLAMA_MODELS`
directory, symlinks the attached read-only blobs, and copies the manifest
instead of calling `client.pull`.

CLI dataset creation was verified with the private smoke dataset:
<https://www.kaggle.com/datasets/devabhirupdas/kaggle-llm-cache-smoke>.

Store native runtime files and model blobs, not a Docker image: Kaggle kernels
cannot replace their base image or run a privileged Docker daemon. A vLLM
wheelhouse is possible but tightly coupled to Kaggle's Python, PyTorch, CUDA,
and vLLM versions; vLLM also needs Transformers-format weights rather than
Ollama blobs.

## Current limitations

- A fresh kernel still installs Ollama and loads the cached model onto the GPU.
- Kaggle CLI launches a committed batch kernel, not an interactive draft.
- The documented CLI can inspect status/output but does not provide a reliable
  command for cancelling a live batch kernel.
- `gemma3n:e4b` handled text but rejected image input in testing.
- Kaggle and ngrok availability, quotas, and policies still apply.
