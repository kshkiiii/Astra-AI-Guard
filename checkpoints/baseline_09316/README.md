# ИИ-сторож

Python-MVP кейса «Астра»: обработка текста, модель, YAML-политика и CLI.

## Рабочие версии

| Версия | Где находится | Score | Block recall high/critical |
| --- | --- | ---: | ---: |
| Текущая: TF-IDF + E5, порог 0.40 | `artifacts/model.joblib` | 0.9316 | 94.44% |
| Предыдущая: binary_risk, порог 0.47 | `checkpoints/baseline_09175/` | 0.9175 | 94.44% |
| Более ранняя: binary, порог 0.35 | `checkpoints/baseline_08881/` | 0.8881 | 91.67% |

Предыдущая версия сохранена целиком: веса, код, конфиг, политика, данные,
контрольные суммы. Показатели относятся к известному локальному test из
95 примеров, который уже использовался при разработке. Это не оценка
качества на скрытом наборе.

## Команды PowerShell из папки проекта

Окружение `.venv` и веса уже готовы; активация окружения не требуется.

```powershell
# Тест текущей модели
.\.venv\Scripts\python.exe check.py --split test

# Валидация текущей модели
.\.venv\Scripts\python.exe check.py --split validation

# Тест сохранённой предыдущей версии
.\.venv\Scripts\python.exe checkpoints/baseline_09175/check.py --split test

# Обучение: перезаписывает только текущие веса, не сохранённую версию
.\.venv\Scripts\python.exe train.py

# Один запрос
.\.venv\Scripts\python.exe check.py --text "Перезапусти payments-api в production" --role sre

# Проверки кода
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Для установки с нуля рекомендуется Python 3.12, минимум 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
```

После установки обучение и проверка не требуют сети.

## Основные файлы

| Путь | Назначение |
| --- | --- |
| `text_processing.py` | Нормализация, история, вложения, окна и декодирование для анализа |
| `model.py` | TF-IDF, обучаемые головы, объединение оценок, сохранение/загрузка |
| `semantic_features.py` | Замороженный локальный E5 через ONNX на CPU |
| `policy.py` | Распознавание операций и проверка полномочий по YAML |
| `train.py` | Фиксированное разбиение и обучение |
| `check.py` | Проверка запросов, отчёты и подбор порога на validation |
| `config.yaml` | Параметры и пути текущей версии |
| `tests/` | Рабочие проверки кода, политики и разделения данных |
| `public_dataset/` | Неизменённые материалы организаторов |
| `additional_datasets/` | Подготовленные источники, переводы и адаптация |
| `artifacts/splits/` | train303 / validation102 / test95 и manifest |
| `reports/validation/`, `reports/test/` | Актуальные метрики, предсказания, детали и ошибки |
| `research/round2/` | Скрипты нового исследования и отдельный альтернативный вариант |

Предыдущие рабочие версии сохранены целиком в checkpoints. Краткое сравнение
исследования — `reports/research_round2.json`, описание — `research/round2/README.md`.

## Модель и параметры

Три головы логистической регрессии на объединении TF-IDF и признаков E5
оценивают атаку, категорию и риск. E5 заморожен, обучаются только головы.
Текущий режим использует `sqrt(P(attack) * (1 - P(risk=none)))`, сравнивая
оценку с порогом 0.40. Дополнительная нормализация leetspeak сохраняет
исходный текст. Замеры ядра на CPU: p95 около 30–35 мс, без HTTP.
Поле `attack_probability` в этом режиме содержит комбинированную оценку,
а не калиброванную вероятность. Старый режим `binary` использует только
голову атаки. Полномочия отдельно читаются из YAML-политики организатора.

В конфиге доступны `C`, `class_weight`, `max_iter`, n-граммы, `min_df`,
пределы и веса признаков, `char_analyzer`, `score_mode`, `attack_threshold`
и параметры текста. `feature_mode` выбирает tfidf/semantic/hybrid/ensemble.
Рабочие параметры: hybrid, C=10, semantic_weight=1, normalize_leetspeak=true.
Изменения обработки, признаков и режима оценки
применяются после обучения. Порог CLI можно менять без переобучения.

```powershell
.\.venv\Scripts\python.exe check.py --split validation --tune-threshold
```

Подбор сохраняет `threshold_search.json` и не меняет конфиг автоматически.
Он разрешён только на validation текущей модели. Подбор на test не выполняется.
ID, метки, имена, tags и difficulty в признаки не подаются.

Дополнительные данные сохранены, но сейчас отключены: `extra_sources: []`.
Для подключения задайте `extra_sources: [additional_datasets/adapted.jsonl]`.
Описание источников и подготовки — в [additional_datasets/README.md](additional_datasets/README.md).

`--input request.json` принимает `user_prompt` и `context` с доверенными
`target_agent`, `user_role`, историей и текстом вложений. Команды не
исполняются, запросы не передаются агенту. API и интерфейс пока не добавлены.

Остаются ложные блокировки, пропуски и ограниченная поддержка обфускаций.
Для следующей независимой оценки нужны новые семейства запросов.

`additional_datasets/control_v1.jsonl` — 48 авторских диагностических
сценариев, не независимая экспертная разметка. Они запрещены для обучения.
Текущий Score на них 0.7143 против 0.6789 у предыдущей версии.
`contrastive_v1.jsonl` — отдельные 48 парных обучающих примеров; они сейчас
не включены в активное обучение. Оба файла не изменяют данные организаторов.

Для переноса текущей версии копируйте также `semantic_features.py` и
`artifacts/encoders/multilingual-e5-small/` (ONNX, токенизатор, карточка и manifest).
При отсутствии файлов восстановите их до автономного запуска:

```powershell
.\.venv\Scripts\python.exe research/round2/download_encoder.py
```

Альтернатива с большим Score и меньшим recall: 0.9428 / 91.67%, порог 0.58.
Она проверяется отдельно, без замены текущей модели:

```powershell
.\.venv\Scripts\python.exe check.py --config research/round2/alternative/config.yaml --split test
```
