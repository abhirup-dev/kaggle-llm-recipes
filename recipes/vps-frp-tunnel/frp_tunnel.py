#!/usr/bin/env python3
"""Run an authenticated FRP TCP tunnel for any local TCP service."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import socket
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass

FRP_VERSION = "0.69.0"
FRP_ARCHIVE = f"frp_{FRP_VERSION}_linux_amd64.tar.gz"
FRP_URL = f"https://github.com/fatedier/frp/releases/download/v{FRP_VERSION}/{FRP_ARCHIVE}"
FRP_SHA256 = "6b90d1cd28fc661f170c0de90dde03d2c63e4fd7ce0ae2da2ca1c28014b8146e"


@dataclass(frozen=True)
class TunnelConfig:
    server_addr: str
    server_port: int
    auth_token: str
    proxy_name: str
    local_addr: str
    local_port: int
    remote_port: int

    @classmethod
    def from_env(cls) -> "TunnelConfig":
        config = cls(
            server_addr=os.environ["FRP_SERVER_ADDR"],
            server_port=int(os.environ.get("FRP_SERVER_PORT", "7443")),
            auth_token=os.environ["FRP_AUTH_TOKEN"],
            proxy_name=os.environ["FRP_PROXY_NAME"],
            local_addr=os.environ.get("FRP_LOCAL_ADDR", "127.0.0.1"),
            local_port=int(os.environ["FRP_LOCAL_PORT"]),
            remote_port=int(os.environ["FRP_REMOTE_PORT"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.server_addr or not self.auth_token:
            raise ValueError("FRP server address and token must not be empty")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.proxy_name):
            raise ValueError("FRP proxy name must contain only letters, numbers, dot, dash, or underscore")
        for name, port in (
            ("server", self.server_port),
            ("local", self.local_port),
            ("remote", self.remote_port),
        ):
            if not 1 <= port <= 65535:
                raise ValueError(f"{name} port must be between 1 and 65535")
    def toml(self, token_file: pathlib.Path) -> str:
        return f"""\
serverAddr = "{self.server_addr}"
serverPort = {self.server_port}
loginFailExit = false
auth.method = "token"
auth.tokenSource.type = "file"
auth.tokenSource.file.path = "{token_file}"
transport.protocol = "tcp"
transport.tls.enable = true
transport.poolCount = 2

[[proxies]]
name = "{self.proxy_name}"
type = "tcp"
localIP = "{self.local_addr}"
localPort = {self.local_port}
remotePort = {self.remote_port}
healthCheck.type = "tcp"
healthCheck.intervalSeconds = 10
healthCheck.timeoutSeconds = 3
healthCheck.maxFailed = 3
"""


def verify_public_ingress(config: TunnelConfig, timeout: float = 8) -> None:
    with socket.create_connection((config.server_addr, config.server_port), timeout=timeout):
        pass


def install_frpc(root: pathlib.Path = pathlib.Path("/tmp/frp")) -> pathlib.Path:
    binary = root / f"frp_{FRP_VERSION}_linux_amd64" / "frpc"
    if binary.is_file():
        return binary
    root.mkdir(parents=True, exist_ok=True)
    archive = root / FRP_ARCHIVE
    urllib.request.urlretrieve(FRP_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != FRP_SHA256:
        raise RuntimeError(f"FRP checksum mismatch: {digest}")
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root)
    if not binary.is_file():
        raise FileNotFoundError(binary)
    binary.chmod(0o755)
    return binary


def run(config: TunnelConfig) -> int:
    verify_public_ingress(config)
    frpc = install_frpc()
    with tempfile.TemporaryDirectory(prefix="frpc-") as temp:
        temp_path = pathlib.Path(temp)
        token_file = temp_path / "token"
        token_file.write_text(config.auth_token)
        token_file.chmod(0o600)
        config_file = temp_path / "frpc.toml"
        config_file.write_text(config.toml(token_file))
        config_file.chmod(0o600)
        subprocess.run([frpc, "verify", "-c", config_file], check=True)
        return subprocess.call([frpc, "-c", config_file])


def self_test() -> None:
    config = TunnelConfig(
        "vps.example",
        7443,
        "secret",
        "service-1",
        "127.0.0.1",
        8080,
        19101,
    )
    config.validate()
    rendered = config.toml(pathlib.Path("/tmp/token"))
    assert 'type = "tcp"' in rendered
    assert "remotePort = 19101" in rendered
    assert "secret" not in rendered
    try:
        TunnelConfig("", 0, "", "bad name", "127.0.0.1", 0, 0).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid configuration was accepted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        raise SystemExit(run(TunnelConfig.from_env()))
