import hashlib
import json
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DESTINATION = ROOT / "artifacts/encoders/multilingual-e5-small"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cert = Path(sys.prefix) / "Lib/site-packages/pip/_vendor/certifi/cacert.pem"
    tls = ssl.create_default_context(cafile=str(cert) if cert.exists() else None)
    manifest = {"repo": REPO, "revision": REVISION, "license": "MIT", "files": []}
    for name in ("onnx/model_qint8_avx512_vnni.onnx", "onnx/tokenizer.json", "README.md"):
        path = DESTINATION / Path(name).name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, context=tls, timeout=120) as response, partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            partial.replace(path)
        manifest["files"].append({"name": path.name, "url": url, "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"Доступен {path.name}: {path.stat().st_size} байт", flush=True)
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
