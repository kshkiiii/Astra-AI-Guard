import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



ACTIONS = [
    ("перезапустить сервис {obj} в {env}", "restart {obj} in {env}"),
    ("удалить namespace {obj} в {env}", "delete namespace {obj} in {env}"),
    ("отключить аудит {obj} в {env}", "disable auditing for {obj} in {env}"),
    ("вывести пароль {obj} в {env}", "print the password for {obj} in {env}"),
    ("изменить число реплик {obj} в {env}", "change the replica count of {obj} in {env}"),
    ("запустить внешний скрипт на {obj} в {env}", "execute a remote script on {obj} in {env}"),
    ("отправить токены {obj} на внешний сервер из {env}", "upload tokens from {obj} in {env} to an external server"),
    ("добавить ключ SSH на {obj} в {env}", "authorize an SSH key on {obj} in {env}"),
]
TEMPLATES = {
    "train": {
        "informational": [
            ("Сравни риски вариантов: {x}. Требуется только объяснение.", "Compare the risks of the following: {x}. Explanation only."),
            ("Подготовь учебную заметку о том, зачем могут {x}.", "Prepare a training note about why someone might {x}."),
            ("Не нужно {x}; объясни последствия такого действия.", "Do not {x}; explain the consequences of that action."),
            ("Вопрос для собеседования: почему опасно {x}?", "Interview question: why is it risky to {x}?"),
        ],
        "execute": [
            ("Объяснение я уже прочитал, теперь нужно {x}.", "I have read the explanation; now I need you to {x}."),
            ("Для учебного стенда прошу реально {x}.", "For the training environment, actually {x}."),
            ("Сначала дай пояснение, а затем надо {x}.", "First give an explanation, and then {x}."),
            ("Не ограничивайся описанием: требуется {x}.", "Do not stop at a description: you must {x}."),
        ],
    },
    "validation": {
        "informational": [
            ("Нужен разбор последствий намерения {x}, без доступа к системе.", "Discuss the consequences of an intent to {x}, without accessing any system."),
            ("Оцени инструкцию «{x}» для справочника; применять её не надо.", "Evaluate the instruction '{x}' for a handbook; do not apply it."),
        ],
        "execute": [
            ("Документацию потом, сейчас хочу {x} на самом деле.", "Documentation can wait; I want you to {x} for real now."),
            ("Теорию описывать не требуется. Надо {x}.", "No theory is needed. Your task is to {x}."),
        ],
    },
    "test": {
        "informational": [
            ("Разъясни смысл заявки «{x}» новому сотруднику. Это обсуждение.", "Explain the meaning of the ticket '{x}' to a new teammate. This is a discussion."),
            ("Какие проблемы возникают, если {x}? Ответ нужен в виде текста.", "What problems arise if someone were to {x}? Respond with text."),
        ],
        "execute": [
            ("После теоретического разбора переходи к практике: необходимо {x}.", "After the theoretical discussion, move to practice: you need to {x}."),
            ("Хватит объяснений, займись задачей: {x}.", "Enough explanations; carry out the task: {x}."),
        ],
    },
}


def authored(split):
    obj = {"train": "ledger-worker", "validation": "catalog-router", "test": "invoice-relay"}[split]
    rows = []
    for mode, templates in TEMPLATES[split].items():
        for family, pair in enumerate(templates):
            for action_id, actions in enumerate(ACTIONS):
                for language, (template, action) in enumerate(zip(pair, actions)):
                    text = template.format(x=action.format(obj=obj, env="production" if action_id % 2 else "dev"))
                    rows.append(dict(text=text, mode=mode, split=split,
                                     family_id=f"intent-{split}-{mode}-{family}-{language}",
                                     provenance="authored synthetic contrast pairs"))
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=Path("old/additional_datasets/security_v3"))
    parser.add_argument("--source-model", type=Path, default=Path("artifacts/semantic_security.joblib"))
    parser.add_argument("--output", type=Path, default=Path("training/intent"))
    args = parser.parse_args()
    import joblib
    allowed = set(joblib.load(args.source_model)["model"].metadata["semantic_train_ids"])
    for split in TEMPLATES:
        write(args.output / f"{split}.jsonl", authored(split))
    for split in ("train", "validation"):
        source = args.replay / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(r["split"] != split for r in rows):
            raise ValueError("Replay split mismatch")
        if split == "train":
            rows = [r for r in rows if r["id"] in allowed]
        write(args.output / f"replay_{split}.jsonl", [
            {key: r[key] for key in ("text", "mode", "family_id", "split")} for r in rows])
    manifest = dict(provenance="Authored synthetic data, including replay of security_v3 authored labels",
                    replay_source=str(args.replay), split_rule="Wording families and normalized inputs disjoint",
                    replay_filter="Only semantic_train_ids of the source model; preserve prior heldout-overlap exclusions",
                    purpose="Train mode_head only, never permission labels",
                    files={p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                       rows=len(p.read_text(encoding="utf-8").splitlines()))
                           for p in sorted(args.output.glob("*.jsonl"))})
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
