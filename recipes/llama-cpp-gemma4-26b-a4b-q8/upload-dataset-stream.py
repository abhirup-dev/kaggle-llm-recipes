import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import requests

try:
    from kaggle.api.kaggle_api_extended import KaggleApi, ResumableUploadResult
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    from kaggle.api.kaggle_api_extended import KaggleApi, ResumableUploadResult


MODEL_FILE = "gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
MODEL_URL = (
    "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/"
    + MODEL_FILE
)
EXPECTED_BYTES = 27_636_232_928
EXPECTED_SHA256 = "d4bf9791d727d7b88aeea89aba309c68086a4d51cf337047c4e51dde7e243058"
HF_TOKEN = os.environ.get("HF_TOKEN", "__HF_TOKEN__")
KAGGLE_API_TOKEN = os.environ.get("KAGGLE_API_TOKEN", "__KAGGLE_API_TOKEN__")
os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
UPLOAD_ROOT = (
    pathlib.Path("/kaggle/working/.upload-cache")
    if pathlib.Path("/kaggle/working").is_dir()
    else pathlib.Path(__file__).resolve().parents[2] / ".upload-cache"
)


class ProgressReader:
    def __init__(self, source, start):
        self.source = source
        self.total = start
        self.next_report = start + 256 * 1024**2
        self.digest = hashlib.sha256() if start == 0 else None
        self.started = time.monotonic()

    def read(self, size=-1):
        chunk = self.source.read(size)
        if chunk:
            self.total += len(chunk)
            if self.digest:
                self.digest.update(chunk)
            if self.total >= self.next_report:
                elapsed = time.monotonic() - self.started
                print(
                    f"STREAM_UPLOAD {self.total / 1024**3:.1f} GiB "
                    f"({self.total / 1024 / 1024 / elapsed:.1f} MiB/s)",
                    flush=True,
                )
                self.next_report += 256 * 1024**2
        return chunk


class RemoteDatasetApi(KaggleApi):
    def upload_complete(self, path, url, quiet, resume=False):
        result = ResumableUploadResult.Incomplete()
        if resume:
            result = self._resume_upload(path, url, EXPECTED_BYTES, quiet)
            if result.result != ResumableUploadResult.INCOMPLETE:
                return result.result

        start = result.start_at
        upload_size = EXPECTED_BYTES - start
        source_headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Accept-Encoding": "identity",
        }
        if start:
            source_headers["Range"] = f"bytes={start}-"

        with requests.get(
            MODEL_URL,
            headers=source_headers,
            stream=True,
            timeout=(30, 600),
        ) as source:
            source.raise_for_status()
            if start and source.status_code != 206:
                raise RuntimeError("Hugging Face ignored the resume range")
            reader = ProgressReader(source.raw, start)
            upload_headers = {"Content-Length": str(upload_size)}
            if start:
                upload_headers["Content-Range"] = (
                    f"bytes {start}-{EXPECTED_BYTES - 1}/{EXPECTED_BYTES}"
                )
            response = requests.put(
                url,
                data=reader,
                headers=upload_headers,
                timeout=(30, 3600),
            )
            if response.status_code in (200, 201):
                if reader.digest and reader.digest.hexdigest() != EXPECTED_SHA256:
                    raise RuntimeError("Streamed model checksum mismatch")
                return ResumableUploadResult.COMPLETE
            if response.status_code == 503:
                return ResumableUploadResult.INCOMPLETE
            print(f"Kaggle upload failed with HTTP {response.status_code}", flush=True)
            return ResumableUploadResult.FAILED


head = requests.head(
    MODEL_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    allow_redirects=False,
    timeout=30,
)
head.raise_for_status()
if int(head.headers["x-linked-size"]) != EXPECTED_BYTES:
    raise RuntimeError("Hugging Face model size changed")
if head.headers["x-linked-etag"].strip('"') != EXPECTED_SHA256:
    raise RuntimeError("Hugging Face model checksum changed")

UPLOAD_ROOT.mkdir(exist_ok=True)
model = UPLOAD_ROOT / MODEL_FILE
model.touch()
os.truncate(model, EXPECTED_BYTES)
(UPLOAD_ROOT / "dataset-metadata.json").write_text(
    json.dumps(
        {
            "id": "devabhirupdas/gemma-4-26b-a4b-ud-q8-gguf",
            "title": "Gemma 4 26B A4B Dynamic Q8 GGUF",
            "licenses": [{"name": "other"}],
            "subtitle": "Unsloth Dynamic 8-bit GGUF for private Kaggle inference",
            "description": (
                "Private cache of the Unsloth Gemma 4 26B A4B "
                "UD-Q8_K_XL GGUF."
            ),
        },
        indent=2,
    )
)
api = RemoteDatasetApi()
api.authenticate()
try:
    result = api.dataset_create_new(str(UPLOAD_ROOT), public=False, quiet=False)
    if result.status.lower() != "ok":
        raise RuntimeError(result.error)
    print(f"DATASET_CREATED {result.url}", flush=True)
finally:
    model.unlink(missing_ok=True)
