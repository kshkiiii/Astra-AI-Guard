import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True, **kwargs)


def verify_artifacts(root=ROOT, fetch=False):
    manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
    for relative, info in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to((root / "artifacts").resolve()):
            raise ValueError("Artifact path escapes artifacts/")
        valid = path.is_file() and path.stat().st_size == info["bytes"]
        if valid:
            with path.open("rb") as stream:
                valid = hashlib.file_digest(stream, "sha256").hexdigest() == info["sha256"]
        if valid:
            continue
        if not fetch:
            raise ValueError(f"Missing or invalid artifact: {relative}")
        print(f"Downloading {relative} ({info['bytes'] // 1024**2} MiB)", flush=True)
        base = manifest["download_base"] if info["lfs"] else manifest["raw_base"]
        url = base + "/" + relative
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(4 * 1024**2):
                    output.write(chunk)
                    if output.tell() > info["bytes"]:
                        raise ValueError("Artifact exceeds manifest size")
            with partial.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if partial.stat().st_size != info["bytes"] or actual != info["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {relative}")
            partial.replace(path)
        except Exception as exc:
            raise RuntimeError(f"Cannot fetch {relative}. Use git lfs pull or the complete release ZIP.") from exc
    print(f"Verified {len(manifest['files'])} model/runtime files.", flush=True)


def environment():
    if sys.version_info[:2] != (3, 12) or platform.system() != "Windows" or sys.maxsize <= 2**32:
        raise RuntimeError("This release requires Windows x64 and Python 3.12 x64.")
    python = ROOT / ".venv/Scripts/python.exe"
    if not python.is_file():
        run([sys.executable, "-m", "venv", ROOT / ".venv"])
    probe = "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))"
    installed = json.loads(subprocess.check_output([str(python), "-X", "utf8", "-c", probe], text=True, encoding="utf-8"))
    requirements = [line.strip().split("==") for line in (ROOT / "requirements.lock.txt").read_text().splitlines() if "==" in line]
    if any(installed.get(name.lower().replace("_", "-")) != version for name, version in requirements):
        args = [python, "-X", "utf8", "-m", "pip", "install", "--disable-pip-version-check", "-r", ROOT / "requirements.lock.txt"]
        for wheels in (ROOT / "wheels", ROOT / "old/delivery/wheels"):
            if wheels.is_dir():
                args += ["--no-index", "--find-links", wheels]
                break
        env = dict(os.environ)
        for name in ("PIP_CERT", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
            if env.get(name) and not Path(env[name]).is_file():
                env.pop(name)
        run(args, env=env)
    return python


def open_when_ready(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(240):
        try:
            with opener.open(f"http://127.0.0.1:{port}/ready", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(f"http://127.0.0.1:{port}/")
                    return
        except OSError:
            pass
        time.sleep(.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    python = environment()
    verify_artifacts(fetch=True)
    if args.check_only:
        print("Ready. Run START.bat to open the demo.")
        return
    if args.test:
        run([python, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", "tests", "-v"])
        run([python, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", "evaluation", "-v"])
        return
    if args.build:
        target = ROOT / "dist/guard-final"
        if target.exists():
            target = ROOT / f"dist/guard-final-{time.time_ns()}"
        command = [python, "-X", "utf8", "-B", "tools/package_release.py", "--output", target, "--zip"]
        for wheels in (ROOT / "wheels", ROOT / "old/delivery/wheels"):
            if wheels.is_dir():
                command += ["--wheels", wheels]
                break
        run(command)
        return
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    env["GUARD_CONFIG"] = str(ROOT / ("config.organizer.yaml" if args.evaluation else "config.yaml"))
    if args.evaluation:
        env.update(GUARD_DEMO_ENABLED="0", GUARD_ADMIN_TOKEN="")
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(args.port,), daemon=True).start()
    print(f"Starting http://127.0.0.1:{args.port}/ ; Ctrl+C stops the service.", flush=True)
    run([python, "-X", "utf8", "-B", "-m", "api", "--host", args.host, "--port", args.port], env=env)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
