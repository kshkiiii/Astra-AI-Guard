import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output.resolve()
    if target.exists() or target == root or not target.is_relative_to(root):
        raise ValueError("Use a new output directory inside the project")
    target.mkdir(parents=True)
    files = [*root.glob("*.py"), *root.glob("*.yaml"), *root.glob("requirements*.txt"), root/"README.md"]
    for name in ["api", "frontend", "artifacts", "tests", "tools", "evaluation", "docs"]:
        for path in (root/name).rglob("*"):
            if path.is_file() and not any(part in {"__pycache__", "runtime", "node_modules"} for part in path.relative_to(root/name).parts) and path.suffix not in {".pyc", ".sqlite3", ".token"}:
                files.append(path)
    for source in files:
        destination = target/source.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    if args.wheels:
        shutil.copytree(args.wheels,target/"wheels")
    if args.report:
        shutil.copytree(args.report,target/"verification")
    (target/"INSTALL.txt").write_text("Windows x64, Python "+sys.version.split()[0]+"\npython -m venv .venv\n.venv\\Scripts\\python -m pip install --no-index --find-links wheels -r requirements.lock.txt\nset GUARD_CONFIG=config.organizer.yaml\n.venv\\Scripts\\python -m api\n\nIf wheels are omitted, install requirements online before inference.\nSee README.md for demo/admin mode and HTTPS hosting preparation.\n",encoding="utf-8")
    manifest = {str(p.relative_to(target)).replace("\\", "/"):hashlib.file_digest(p.open("rb"),"sha256").hexdigest() for p in target.rglob("*") if p.is_file()}
    (target/"SHA256.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print(json.dumps(dict(output=str(target),files=len(manifest),bytes=sum(p.stat().st_size for p in target.rglob("*") if p.is_file())),ensure_ascii=False))


if __name__ == "__main__":
    main()
