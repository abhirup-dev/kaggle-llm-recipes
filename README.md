# Kaggle LLM Recipes

Run Gemma 3n with Ollama on a private Kaggle GPU kernel and expose its
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

The default `e2b-pull` profile downloads the 5.6 GB model directly because it
was the fastest measured cold-start path. Other profiles:

```sh
./start-server e2b       # attached 5.24 GB e2b cache
./start-server e4b       # attached 7.55 GB e4b cache
./start-server e2b-pull  # direct Ollama pull; same as the default
```

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
    "model": "gemma3n:e2b",
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

## Cold-start comparison on Kaggle T4 x2

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
private dataset containing Ollama's manifest and referenced blobs removes the
registry download, but the observed input-staging cost was larger. Keep caches
as an offline/fallback option; direct pull is the recommended fast path.

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

`start-server e4b` and `start-server e2b` attach the matching cache
automatically. The server constructs a writable `OLLAMA_MODELS` directory,
symlinks the attached read-only blobs, and copies the manifest instead of
calling `client.pull`.

CLI dataset creation was verified with the private smoke dataset:
<https://www.kaggle.com/datasets/devabhirupdas/kaggle-llm-cache-smoke>.

Store native runtime files and model blobs, not a Docker image: Kaggle kernels
cannot replace their base image or run a privileged Docker daemon. A vLLM
wheelhouse is possible but tightly coupled to Kaggle's Python, PyTorch, CUDA,
and vLLM versions; vLLM also needs Transformers-format weights rather than
Ollama blobs.

## Current limitations

- A fresh kernel still installs Ollama and loads the cached model onto the GPU.
- Cold-start time varies with Kaggle scheduling and input staging; one run is
  directional rather than a full statistical benchmark.
- Kaggle CLI launches a committed batch kernel, not an interactive draft.
- The documented CLI can inspect status/output but does not provide a reliable
  command for cancelling a live batch kernel.
- `gemma3n:e4b` handled text but rejected image input in testing.
- Kaggle and ngrok availability, quotas, and policies still apply.
