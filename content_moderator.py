
import re
from local_runtime import LocalModelClient, ModelInputTooLong, ModelUnavailable

CATEGORIES = {"Violent", "Non-violent Illegal Acts", "Sexual Content or Sexual Acts", "PII", "Suicide & Self-Harm", "Unethical Acts", "Politically Sensitive Topics", "Copyright Violation", "Jailbreak", "None"}


class ContentModerator:
    def __init__(self, settings):
        self.settings = settings
        self.client = LocalModelClient(settings)
        self.client.ensure_running()
        self.assess("Объясни принцип минимальных привилегий.")

    def assess(self, text):
        if len(text) > self.settings.get("max_input_chars", 16000):
            raise ModelInputTooLong("Content moderation input exceeds the configured limit")
        messages = [dict(role="user", content=text)]
        formatted = self.client.request("/apply-template", dict(messages=messages, add_generation_prompt=True))


        if "<BEGIN SAFETY POLICY>" not in formatted.get("prompt", ""):
            raise ModelUnavailable("Qwen3Guard moderation template is missing")
        tokens = self.client.request("/tokenize", dict(content=formatted["prompt"], add_special=False))["tokens"]
        if len(tokens)+32 > self.client.ctx_size:
            raise ModelInputTooLong("Content moderation context overflow")
        reply = self.client.request("/v1/chat/completions", dict(model=self.client.alias, messages=messages,
            temperature=0, seed=42, max_tokens=32, cache_prompt=True))
        try:
            choice = reply["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Incomplete moderation response")
            content = choice["message"]["content"].strip()
            match = re.fullmatch(r"Safety:\s*(Safe|Unsafe|Controversial)\s+Categories:\s*([^\n]+)", content)
            if not match:
                raise ValueError("Invalid moderation response")
            categories = [v.strip().rstrip(".") for v in match[2].split(",")]
            if not categories or any(v not in CATEGORIES for v in categories):
                raise ValueError("Unknown moderation category")
            if match[1] == "Safe" and categories != ["None"]:
                raise ValueError("Inconsistent moderation response")
            return dict(safety=match[1], categories=categories)
        except (KeyError,IndexError,TypeError,ValueError):
            raise ModelUnavailable("Qwen3Guard returned an invalid safety classification") from None
