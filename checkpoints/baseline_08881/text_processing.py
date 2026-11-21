"""Подготовка только входных полей API. Никакие команды здесь не исполняются."""
from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    source: str
    text: str


@dataclass(frozen=True)
class PreparedInput:
    text: str
    chunks: tuple[str, ...]
    segments: tuple[Segment, ...]


CYR_TO_LAT = str.maketrans("асеорхуікмт", "aceopxyikmt")
LAT_TO_CYR = str.maketrans("aceopxy", "асеорху")


def normalize(text: str, homoglyphs: bool = True) -> str:
    """NFKC, регистр, невидимые разделители; исправление смешанного алфавита."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    if homoglyphs:
        def fix(match: re.Match) -> str:
            token = match.group()
            latin = len(re.findall(r"[a-z]", token))
            cyrillic = len(re.findall(r"[а-яё]", token))
            if latin and cyrillic:
                return token.translate(CYR_TO_LAT if latin >= cyrillic else LAT_TO_CYR)
            return token
        text = re.sub(r"[\w@]+", fix, text)
    return re.sub(r"[ \t]+", " ", text).strip()


def text_variants(text: str, options: dict) -> list[str]:
    """Исходный смысл сохраняется; декодирование лишь добавляет представление."""
    normalized = normalize(text, options.get("normalize_homoglyphs", True))
    variants = [normalized]
    spaced = re.sub(r"(?<!\w)(?:[a-zа-яё]\s+){3,}[a-zа-яё](?!\w)",
                    lambda m: re.sub(r"\s", "", m.group()), normalized)
    if spaced != normalized:
        variants.append(spaced)
    if re.search(r"задом напер[её]д|backwards|reverse (?:this|the text)", normalized):
        payload = text.split(":", 1)[-1]
        variants.append(normalize(payload[::-1]))
    if options.get("decode_base64", True):
        # Ограниченная глубина/размер; ошибочные и бинарные данные игнорируются.
        pending = [text]
        for _ in range(2):
            decoded = []
            for candidate in pending:
                for token in re.findall(r"(?<![\w+/])[A-Za-z0-9+/]{16,}={0,2}(?![\w+/=])", candidate):
                    try:
                        value = base64.b64decode(token + "=" * (-len(token) % 4), validate=True).decode("utf-8")
                        if value and sum(c.isprintable() or c.isspace() for c in value) / len(value) > .95:
                            decoded.append(value)
                    except (ValueError, UnicodeError):
                        continue
            variants.extend(normalize(v) for v in decoded)
            pending = decoded
    return list(dict.fromkeys(variants))


def prepare_input(user_prompt: str, context: dict, options: dict) -> PreparedInput:
    if not isinstance(user_prompt, str) or not isinstance(context, dict):
        raise ValueError("Ожидаются user_prompt: str и context: dict")
    for field in ("user_role", "target_agent"):
        if not isinstance(context.get(field), str) or not context[field]:
            raise ValueError(f"В context обязательно строковое поле {field}")
    segments = [Segment("current_user", user_prompt)]
    history = context.get("history", [])
    attachments = context.get("attachments", [])
    if not isinstance(history, list) or not isinstance(attachments, list):
        raise ValueError("history и attachments должны быть массивами")
    for message in history:
        if (not isinstance(message, dict) or message.get("role") not in ("user", "assistant")
                or not isinstance(message.get("content"), str)):
            raise ValueError("Реплика истории должна содержать role=user|assistant и content: str")
        segments.append(Segment("history_" + message["role"], message["content"]))
    for attachment in attachments:
        if not isinstance(attachment, dict) or not isinstance(attachment.get("content"), str):
            raise ValueError("Вложение должно содержать content: str")
        segments.append(Segment("attachment", attachment["content"]))
    if sum(len(s.text) for s in segments) > options.get("max_input_chars", 200000):
        raise ValueError("Слишком длинный вход: превышен max_input_chars")
    size, overlap = options.get("chunk_size", 3000), options.get("chunk_overlap", 300)
    if not isinstance(size, int) or not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ValueError("Требуется 0 <= chunk_overlap < chunk_size")
    pieces, chunks = [], []
    for segment in segments:
        body = "\n[variant]\n".join(text_variants(segment.text, options))
        marker = f"[{segment.source}]\n"
        pieces.append(marker + body)
        for offset in range(0, max(1, len(body)), size - overlap):
            chunks.append(marker + body[offset:offset + size])
            if offset + size >= len(body):
                break
    full = "\n".join(pieces)
    # Короткий запрос рассматривается целиком, включая диалог. На длинном входе
    # проверяем ещё окна, иначе небольшой вредоносный фрагмент может затеряться.
    views = (full,) if len(segments) == 1 and len(full) <= size else tuple(dict.fromkeys([full, *chunks]))
    return PreparedInput(full, views, tuple(segments))
