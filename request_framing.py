
import re

from text_processing import normalize

QUOTES = re.compile(r'```.*?```|<!--.*?-->|`[^`]*`|«[^»]*»|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\']*\'', re.S)


def framing(text):
    value = QUOTES.sub(" ", normalize(text))
    return re.sub(r"(?m)^\s*(?:user|assistant|system):.*$", " ", value)
