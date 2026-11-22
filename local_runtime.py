
from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class ModelUnavailable(RuntimeError):
    pass


class ModelInputTooLong(ValueError):
    pass


_processes = {}
_startup_lock = threading.Lock()


def _cleanup():
    for process in _processes.values():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


atexit.register(_cleanup)


class LocalModelClient:
    def __init__(self, settings):
        self.settings = settings
        self.alias = "astra-" + settings.get("model_name", Path(settings["model"]).stem)
        self.alias += "-" + Path(settings["model"]).stem
        self.endpoint = settings["endpoint"].rstrip("/")
        endpoint = urllib.parse.urlparse(self.endpoint)
        if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"} or endpoint.username:
            raise ValueError("LLM endpoint must be local HTTP without credentials")
        self.port = endpoint.port or 80
        self.timeout = settings.get("timeout_seconds", 90)
        self.ctx_size = settings.get("context_tokens", 8192)

        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.endpoint + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self.http.open(request, timeout=timeout or self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise ModelUnavailable("Local model request failed") from None

    def ready(self):
        try:
            self.request("/health", timeout=1)
            models = self.request("/v1/models", timeout=1)
            return any(m.get("id") == self.alias for m in models.get("data", []))
        except ModelUnavailable:
            return False

    def ensure_running(self):
        if self.ready():
            return
        with _startup_lock:
            if self.ready():
                return
            binary = Path(self.settings["binary"])
            model = Path(self.settings["model"])
            if not binary.is_file() or not model.is_file():
                raise FileNotFoundError("The configured CPU runtime or moderator model is missing from artifacts/")
            key = (str(binary), str(model), self.endpoint, self.alias)
            process = _processes.get(key)
            if process is None or process.poll() is not None:
                args = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(self.port),
                        "--alias", self.alias, "--ctx-size", str(self.ctx_size), "--parallel", "1",
                        "--threads", str(self.settings.get("threads", 8)), "--threads-batch", str(self.settings.get("threads", 8)),
                        "--n-gpu-layers", "0", "--jinja", "--log-disable", "--cache-ram", "0", "--no-cache-idle-slots"]
                process = subprocess.Popen(args, cwd=str(binary.parent), stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                _processes[key] = process
            deadline = time.monotonic() + self.settings.get("startup_timeout_seconds", 90)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ModelUnavailable("Local model process exited during startup")
                if self.ready():
                    return
                time.sleep(.25)
            raise ModelUnavailable("Local model startup timed out")
