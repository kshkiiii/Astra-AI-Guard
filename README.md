# ИИ-сторож

## Запуск

Python 3.11+ (рекомендуется 3.12), команды из корня проекта в PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check.py --text "Перезапусти payments-api в production" --role sre
```

Готовая модель находится в `artifacts/model.joblib`, локальный энкодер — в `artifacts/encoders/`. При их наличии проверка работает без сети. Если энкодер отсутствует, скачайте его перед запуском:

```powershell
.\.venv\Scripts\python.exe tools/download_encoder.py
```

Обучение, оценка и тесты:

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe check.py --split test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`train.py` создаёт или обновляет модель. Параметры и пути заданы в `config.yaml`, результаты оценки сохраняются в `reports/`.

## HTTP API

Сервис вынесен в отдельную директорию `api/`. Запуск из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m pip install -r api/requirements.txt
.\.venv\Scripts\python.exe -m api
```

Демо-интерфейс — http://127.0.0.1:8000/ (исходники в `frontend/`).
Проверка — `POST http://127.0.0.1:8000/v1/check`, Swagger — http://127.0.0.1:8000/docs.
API использует существующие веса и YAML-политику; модель загружается и прогревается один раз.
В `api/` также находятся отдельные маршруты передачи демо-агенту, явного подтверждения и журнал без исходных текстов.
Контракт, примеры и тесты: [api/README.md](api/README.md).
Работа с интерфейсом: [frontend/README.md](frontend/README.md). Страница и API запускаются одной командой, без Node.js.

## О проекте

ИИ-сторож для кейса «Астра» проверяет запросы к DevOps-агенту и возвращает `allow`, `confirm` или `block`, категорию риска и объяснение. Модель выявляет опасные инструкции, YAML-политика проверяет полномочия пользователя.

## Технологии

- Python, NumPy, SciPy, scikit-learn: TF-IDF и три головы логистической регрессии для атаки, категории и риска.
- multilingual E5, ONNX Runtime, Tokenizers: локальные семантические признаки на CPU.
- PyYAML: правила доступа; joblib: сохранение модели.

## Фичи

- Проверка запроса, истории диалога и текстовых вложений.
- Обнаружение prompt injection и опасных запросов, проверка операций по роли и окружению.
- Нормализация Unicode, смешанных алфавитов, leetspeak и анализ Base64.
- Анализ длинного текста перекрывающимися окнами.
- Изменение YAML-политики без переобучения модели.
- Проверка JSON через `check.py --input request.json`, отчёты по JSONL и подбор порога на validation.

Контекст JSON-запроса содержит `target_agent`, `user_role`, `history` и `attachments`. Поддерживаются роли `developer`, `sre`, `admin`. Команды из запросов не исполняются.
