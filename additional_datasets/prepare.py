from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def download_sources() -> None:
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    cert = Path(sys.prefix) / "Lib/site-packages/pip/_vendor/certifi/cacert.pem"
    tls = ssl.create_default_context(cafile=str(cert) if cert.exists() else None)
    previous_path = ROOT / "downloads.json"
    previous = {e["path"]: e["sha256"] for e in json.loads(previous_path.read_text(encoding="utf-8"))} if previous_path.exists() else {}
    manifest = []
    for source in sources:
        for name in source["files"]:
            target = ROOT / "raw" / source["key"] / name
            url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            relative = str(target.relative_to(ROOT)).replace("\\", "/")
            if target.exists() and relative in previous and hashlib.sha256(target.read_bytes()).hexdigest() != previous[relative]:
                raise ValueError(f"Изменён локальный исходник: {relative}")
            if not target.exists():
                with urllib.request.urlopen(url, context=tls, timeout=120) as response:
                    content = response.read(128 * 1024 * 1024 + 1)
                if len(content) > 128 * 1024 * 1024:
                    raise ValueError(f"Источник слишком велик: {name}")
                target.write_bytes(content)
            manifest.append({"path": relative,
                             "url": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                             "bytes": target.stat().st_size, "license": source["license"]})
            print(f"Скачан/доступен: {source['key']}/{name} ({target.stat().st_size} bytes)", flush=True)
    license_path = ROOT / "LICENSE"
    license_url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
    if not license_path.exists():
        with urllib.request.urlopen(license_url, context=tls, timeout=40) as response:
            license_path.write_bytes(response.read(65536))
    manifest.append({"path": "LICENSE", "url": license_url, "license": "Apache-2.0",
                     "sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(), "bytes": license_path.stat().st_size})
    (ROOT / "downloads.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare() -> None:
    import pyarrow.parquet as pq
    from model import load_config
    from policy import SecurityPolicy
    from text_processing import prepare_input
    from train import write_json, write_jsonl

    config = load_config(ROOT.parent / "config.yaml")
    policy = SecurityPolicy(config["paths"]["policy"])
    sources = {s["key"]: s for s in json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))}
    reviewed = json.loads((ROOT / "reviewed.json").read_text(encoding="utf-8"))
    for entry in json.loads((ROOT / "downloads.json").read_text(encoding="utf-8")):
        if hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Изменён скачанный файл: {entry['path']}")
    raw = {
        "russian_injections": json.loads((ROOT / "raw/russian_injections/dataset.json").read_text(encoding="utf-8")),
        "neuralchemy": pq.read_table(ROOT / "raw/neuralchemy/core/train-00000-of-00001.parquet").to_pylist(),
        "deepset": pq.read_table(ROOT / "raw/deepset/data/train-00000-of-00001-9564e8b05b4757ab.parquet").to_pylist(),
    }
    
    with Path(config["data"]["source"]).open(encoding="utf-8-sig") as stream:
        system_prompt = json.loads(next(stream))["context"].get("system_prompt", "")
    rows = []

    def make_row(key, index, text, lang, category, upstream, translation="original", variant="direct"):
        source = sources[key]
        family = f"{key}:{index}"
        role = ("developer", "sre", "admin")[int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 3]
        context = {"target_agent": "it_ops_agent", "user_role": role, "history": [], "attachments": [],
                   "system_prompt": system_prompt}
        prompt = text
        if category == "indirect_injection":
            prompt = "Кратко перескажи вложение." if lang == "ru" else "Summarize the attachment."
            context["attachments"] = [{"type": "document", "name": "external_sample.txt", "content": text}]
        decision = "allow" if category == "none" else "block"
        risk = "none" if category == "none" else upstream.get("severity") or "high"
        if category == "none":
            
            result = policy.evaluate(prompt, context, config["text"])
            decision, category, risk = result["decision"], result["risk_category"], result["risk_level"]
        row = {"id": f"extra-{key}-{index}-{lang}-{variant}", "user_prompt": prompt, "context": context,
               "is_dangerous": int(decision == "block"), "risk_level": risk,
               "risk_category": category, "expected_decision": decision, "lang": lang,
               "difficulty": "hard", "tags": ["external_training_only"],
               "provenance": {"dataset": source["repo"], "revision": source["revision"],
                              "source_split": "train", "row_index": index, "license": source["license"],
                              "family_id": family, "translation": translation, "adaptation": variant,
                              "upstream_label": upstream.get("label", upstream.get("class")),
                              "upstream_category": upstream.get("category", upstream.get("class")),
                              "upstream_group_id": upstream.get("group_id"), "upstream_source": upstream.get("source")}}
        prepare_input(row["user_prompt"], context, config["text"])
        return row

    for key, selections in reviewed.items():
        for selection in selections:
            index, category = selection[:2]
            original = raw[key][index]
            if key == "russian_injections":
                if original["source"] != "openai_synthetic":
                    raise ValueError("Для русского набора разрешена только проверенная synthetic-подвыборка")
                pair = [original["prompt_en"], original["prompt_ru"]]
                
                ru = max(pair, key=lambda s: len(re.findall(r"[а-яё]", s.casefold())))
                ru = re.sub(r"^Вот перевод на русский язык[^:]*:\\n\\n", "", ru)
                rows.append(make_row(key, index, ru, "ru", category, original, "upstream_ru_selected_by_script"))
            else:
                if key == "neuralchemy" and original["source"] != "original":
                    raise ValueError("Не разрешено смешивать лицензии upstream-компиляций")
                if key == "deepset" and original["label"] != 0:
                    raise ValueError("Из deepset выбраны только проверенные легитимные запросы")
                rows.append(make_row(key, index, original["text"], "en", category, original))
                rows.append(make_row(key, index, selection[2], "ru", category, original, "translated_and_reviewed_by_codex"))

    
    
    templates = {
        "read_logs": [("Покажи логи сервиса {resource} в {env}.", "Show logs for {resource} in {env}."),
                      ("Прочитай последние логи {resource} в {env}.", "Read the latest logs of {resource} in {env}."),
                      ("Выведи логи {resource} в {env}.", "Get logs for {resource} in {env}.")],
        "restart_service": [("Перезапусти сервис {resource} в {env}.", "Restart service {resource} in {env}."),
                            ("Перезапусти {resource} в {env}, пожалуйста.", "Please restart {resource} in {env}."),
                            ("Выполни перезапуск {resource} в {env}.", "Execute a restart of {resource} in {env}.")],
        "run_shell": [("Выполни df -h на {resource} в {env}.", "Run df -h on {resource} in {env}."),
                      ("Выполни uptime на {resource} в {env}.", "Execute uptime on {resource} in {env}."),
                      ("Выполни команду free -m на {resource} в {env}.", "Run the command free -m on {resource} in {env}.")],
        "delete_resource": [("Удали ресурс {resource} в {env}.", "Delete resource {resource} in {env}."),
                            ("Удали volume {resource} в {env}.", "Remove volume {resource} in {env}."),
                            ("Удали namespace {resource} в {env}.", "Delete namespace {resource} in {env}.")],
    }
    for tool, pairs in templates.items():
        for env in ("dev", "production"):
            for variant, pair in enumerate(pairs):
                family = f"devops:{tool}:{env}:{variant}"
                for role in ("developer", "sre", "admin"):
                    for lang, template in zip(("ru", "en"), pair):
                        text = template.format(resource="sandbox-worker", env=env)
                        row = make_row("neuralchemy", 3, text, lang, "none", raw["neuralchemy"][3])
                        row["context"]["user_role"] = role
                        result = policy.evaluate(text, row["context"], config["text"])
                        row.update(id=f"extra-devops-{tool}-{env}-{variant}-{role}-{lang}",
                                   expected_decision=result["decision"], risk_category=result["risk_category"],
                                   risk_level=result["risk_level"], is_dangerous=int(result["decision"] == "block"))
                        row["provenance"].update(family_id=family, adaptation="generated_role_environment_counterfactual",
                                                 translation="generated_bilingual_template", policy_version=policy.version)
                        rows.append(row)
    write_jsonl(ROOT / "adapted.jsonl", rows)
    summary = {"rows": len(rows), "languages": dict(Counter(r["lang"] for r in rows)),
               "decisions": dict(Counter(r["expected_decision"] for r in rows)),
               "categories": dict(Counter(r["risk_category"] for r in rows)),
               "reviewed_source_rows": {k: len(v) for k, v in reviewed.items()},
               "adaptations": dict(Counter(r["provenance"]["adaptation"] for r in rows)),
               "note": "Только train; окончательное исключение дублей с public/validation/test выполняет train.py."}
    write_json(ROOT / "preparation_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description='Загрузка закреплённых версий открытых данных и адаптация для train.\n\nДанные читаются как JSON/Parquet, никакой код из репозиториев не запускается.')
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.download or args.download_only:
        download_sources()
    if not args.download_only:
        prepare()


if __name__ == "__main__":
    main()
