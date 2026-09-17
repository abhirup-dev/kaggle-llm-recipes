#!/usr/bin/env python3
"""Structured worker telemetry and a Fluent Bit log shipper."""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
import threading
import time

WORK_DIR = pathlib.Path("/kaggle/working")
EVENTS_PATH = WORK_DIR / "worker-events.jsonl"
LLAMA_LOG_PATH = WORK_DIR / "llama-server.log"
FLUENT_BIT = pathlib.Path("/opt/fluent-bit/bin/fluent-bit")


def install_observability(with_log_relay: bool) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "psutil==7.0.0",
            "nvidia-ml-py==13.580.82",
            "python-json-logger==3.3.0",
        ],
        check=True,
    )
    if not with_log_relay or FLUENT_BIT.is_file():
        return

    release = {}
    for line in pathlib.Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip('"')
    codename = release.get("VERSION_CODENAME", "")
    if codename not in {"focal", "jammy", "noble"}:
        raise RuntimeError(f"unsupported Fluent Bit Ubuntu release: {codename!r}")

    key = pathlib.Path("/tmp/fluentbit.key")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "https://packages.fluentbit.io/fluentbit.key",
            "--output",
            key,
        ],
        check=True,
    )
    subprocess.run(
        [
            "gpg",
            "--batch",
            "--yes",
            "--dearmor",
            "--output",
            "/usr/share/keyrings/fluentbit-keyring.gpg",
            key,
        ],
        check=True,
    )
    pathlib.Path("/etc/apt/sources.list.d/fluent-bit.list").write_text(
        "deb [signed-by=/usr/share/keyrings/fluentbit-keyring.gpg] "
        f"https://packages.fluentbit.io/ubuntu/{codename} {codename} main\n"
    )
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "--no-install-recommends", "fluent-bit"],
        check=True,
    )


def configure_logger() -> logging.Logger:
    from pythonjsonlogger.json import JsonFormatter

    logger = logging.getLogger("kaggle-worker")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(message)s %(event)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    for handler in (
        logging.FileHandler(EVENTS_PATH),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def emit(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(event, extra={"event": event, **fields})


def fluent_bit_config(worker_id: str, relay_host: str, relay_port: int) -> str:
    return f"""\
[SERVICE]
    Flush                     1
    Grace                     5
    Log_Level                 info
    Parsers_File              {WORK_DIR / "fluent-bit-parsers.conf"}
    storage.path              {WORK_DIR / "fluent-bit-storage"}
    storage.sync              normal
    storage.checksum          on
    storage.max_chunks_up     8

[INPUT]
    Name                      tail
    Tag                       kaggle.events
    Path                      {EVENTS_PATH}
    Parser                    json
    DB                        {WORK_DIR / "events-tail.db"}
    DB.Sync                   normal
    Read_from_Head            true
    Refresh_Interval          1
    Skip_Long_Lines           true
    storage.type              filesystem

[INPUT]
    Name                      tail
    Tag                       kaggle.llama
    Path                      {LLAMA_LOG_PATH}
    DB                        {WORK_DIR / "llama-tail.db"}
    DB.Sync                   normal
    Read_from_Head            true
    Refresh_Interval          1
    Skip_Long_Lines           true
    storage.type              filesystem

[FILTER]
    Name                      modify
    Match                     *
    Add                       worker_id {worker_id}

[OUTPUT]
    Name                      forward
    Match                     *
    Host                      {relay_host}
    Port                      {relay_port}
    Shared_Key                ${{LOG_RELAY_SHARED_KEY}}
    Self_Hostname             {worker_id}
    Require_ack_response      true
    Compress                  gzip
    Retry_Limit               false
    tls                       on
    tls.verify                on
    tls.verify_hostname       off
    tls.ca_file               {WORK_DIR / "log-relay-ca.crt"}
"""


def start_fluent_bit(worker_id: str) -> tuple[subprocess.Popen, object]:
    relay_port = int(os.environ.get("LOG_RELAY_PORT", "24224"))
    pathlib.Path(WORK_DIR / "log-relay-ca.crt").write_text(os.environ["LOG_RELAY_CA_PEM"])
    pathlib.Path(WORK_DIR / "fluent-bit-parsers.conf").write_text(
        "[PARSER]\n"
        "    Name        json\n"
        "    Format      json\n"
        "    Time_Key    timestamp\n"
        "    Time_Keep   on\n"
    )
    pathlib.Path(WORK_DIR / "fluent-bit.conf").write_text(
        fluent_bit_config(worker_id, os.environ["LOG_RELAY_HOST"], relay_port)
    )
    LLAMA_LOG_PATH.touch()
    fluent_log = (WORK_DIR / "fluent-bit.log").open("w", buffering=1)
    process = subprocess.Popen(
        [FLUENT_BIT, "--config", WORK_DIR / "fluent-bit.conf"],
        stdout=fluent_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError((WORK_DIR / "fluent-bit.log").read_text())
    return process, fluent_log


def start_resource_sampler(
    logger: logging.Logger,
    process_pid: int,
    interval: int = 30,
) -> threading.Event:
    import psutil
    import pynvml

    stop = threading.Event()
    process = psutil.Process(process_pid)
    psutil.cpu_percent()
    process.cpu_percent()
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]

    def sample() -> None:
        while not stop.is_set():
            try:
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                process_memory = process.memory_info()
                gpus = []
                for index, handle in enumerate(handles):
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu = {
                        "index": index,
                        "name": pynvml.nvmlDeviceGetName(handle),
                        "gpu_usage_pct": utilization.gpu,
                        "memory_usage_pct": utilization.memory,
                        "memory_total_bytes": gpu_memory.total,
                        "memory_used_bytes": gpu_memory.used,
                        "memory_free_bytes": gpu_memory.free,
                        "temperature_c": pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        ),
                    }
                    try:
                        gpu["power_w"] = round(
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 2
                        )
                    except pynvml.NVMLError:
                        pass
                    gpus.append(gpu)
                emit(
                    logger,
                    "resource_sample",
                    cpu={
                        "usage_pct": psutil.cpu_percent(),
                        "count": psutil.cpu_count(),
                        "load": os.getloadavg(),
                    },
                    memory={
                        "total_bytes": memory.total,
                        "available_bytes": memory.available,
                        "used_bytes": memory.used,
                        "usage_pct": memory.percent,
                        "swap_total_bytes": swap.total,
                        "swap_used_bytes": swap.used,
                    },
                    server_process={
                        "pid": process_pid,
                        "cpu_usage_pct": process.cpu_percent(),
                        "rss_bytes": process_memory.rss,
                        "vms_bytes": process_memory.vms,
                    },
                    gpus=gpus,
                )
            except Exception:
                logger.exception(
                    "resource_sample_failed",
                    extra={"event": "resource_sample_failed"},
                )
            stop.wait(interval)

    threading.Thread(target=sample, name="resource-sampler", daemon=True).start()
    return stop


def self_test() -> None:
    config = fluent_bit_config("worker-1", "vps.example", 24224)
    assert "Name                      tail" in config
    assert "Require_ack_response      true" in config
    assert "${LOG_RELAY_SHARED_KEY}" in config
    assert "secret" not in config


if __name__ == "__main__":
    self_test()
