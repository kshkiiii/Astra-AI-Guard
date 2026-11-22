# ИИ-сторож
## Команда Газмяс
Локальная защита DevOps-ассистента: проверяет запросы, историю и вложения с учётом роли пользователя и YAML-политики. Возвращает `allow`, `confirm` или `block` с объяснением. Работает на CPU; после установки интернет не требуется.

## Быстрая установка

Требуются **Windows x64**, **Python 3.12 x64** и **Git LFS**.

```powershell
git lfs install
git clone https://github.com/kshkiiii/Astra-AI-Guard.git
cd Astra-AI-Guard
.\START.bat
```

Скрипт установит зависимости, проверит веса и откроет интерфейс. Первый запуск из Git требует интернета.

Из готового комплекта: распакуйте ZIP и дважды нажмите **START.bat**.

[Интерфейс](http://127.0.0.1:8000/) · [Документация API](http://127.0.0.1:8000/docs) · [хост](https://posing-bryant-salaries-awarded.trycloudflare.com/)

## Стек технологий

- **Backend:** Python, FastAPI, Uvicorn, Pydantic.
- **ML:** scikit-learn, ONNX Runtime, E5-small, MiniLM, Qwen3Guard 0.6B, llama.cpp.
- **Данные и политика:** SQLite, YAML.
- **Frontend:** HTML, CSS, JavaScript.
