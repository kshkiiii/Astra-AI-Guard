import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.launch import verify_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output.resolve()
    if target.exists() or target == root or not target.is_relative_to(root):
        raise ValueError("Use a new output directory inside the project")
    verify_artifacts(root)
    if args.wheels and not args.wheels.is_dir():
        raise ValueError("Wheel directory not found")
    archive = target.with_suffix(".zip")
    if args.zip and archive.exists():
        raise ValueError("Release ZIP already exists")
    target.mkdir(parents=True)
    files = [*root.glob("*.py"), *root.glob("*.yaml"), *root.glob("*.bat"), *root.glob("requirements*.txt"), root/"README.md"]
    artifacts = json.loads((root/"artifacts/manifest.json").read_text(encoding="utf-8"))
    files += [root/"artifacts/manifest.json", *[root/path for path in artifacts["files"]]]
    for name in ["api", "frontend", "tests", "tools", "training", "evaluation", "docs"]:
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
    (target/"INSTALL.txt").write_text("Windows x64, Python 3.12 x64.\nUnzip to a writable folder. Double-click START.bat.\nSTART.bat --test : run tests\nSTART.bat --evaluation --host 0.0.0.0 : organizer endpoint\nSTART.bat --check-only : verify models and dependencies\n\nIncluded wheels install offline. If wheels are absent, first setup needs internet.\nInference is offline. API: http://127.0.0.1:8000/docs\nCtrl+C stops the service. See README.md.\n",encoding="utf-8")
    manifest = {str(p.relative_to(target)).replace("\\", "/"):hashlib.file_digest(p.open("rb"),"sha256").hexdigest() for p in target.rglob("*") if p.is_file()}
    (target/"SHA256.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    if args.zip:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
            for path in sorted(target.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(target))
    print(json.dumps(dict(output=str(target),archive=str(archive) if args.zip else None,files=len(manifest),bytes=sum(p.stat().st_size for p in target.rglob("*") if p.is_file())),ensure_ascii=False))


if __name__ == "__main__":
    main()
