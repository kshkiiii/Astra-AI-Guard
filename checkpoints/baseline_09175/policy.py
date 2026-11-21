"""Извлечение DevOps-действий и детерминированная проверка YAML-политики.

Это ограниченный baseline, а не shell-парсер. Неоднозначность не даёт новых прав.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import yaml

from text_processing import normalize, text_variants


def educational(text: str) -> bool:
    # Информационной должна быть сама просьба, а не произвольная приписка
    # «и объясни почему» после команды. Вежливые вводные не меняют намерение.
    body = re.sub(
        r"^(?:(?:добрый день|здравствуйте|привет|слушай|срочно|вопрос|пожалуйста|"
        r"hello|hey|hi|urgent|please)\b[\s,.!:;—-]*)+", "", text)
    return bool(re.match(
        r"(?:сначала\s+)?(?:объясни|расскажи.{0,20}(?:как работает|что такое)|"
        r"почему\b|что (?:означает|такое)|как\b|разбер[иё]|для (?:онбординга|учебного)|"
        r"без (?:выполнения|технических))|"
        r"(?:first\s+)?(?:explain\b|what (?:is|does)|why\b|how (?:does|to prevent)|"
        r"without (?:executing|technical))", body))


def action_text(user_prompt: str, context: dict, options: dict) -> str:
    text = "\n".join(text_variants(user_prompt, options))
    # История помогает разрешить «сделай это», но не меняет доверенную роль.
    if re.search(r"сделай (?:это|так)|выполни (?:это|е[её])|продолж|do it|execute it|continue", text):
        history = [m["content"] for m in context.get("history", []) if m["role"] == "user"]
        text = "\n".join([*(normalize(s) for s in history[-3:]), text])
    return text


class SecurityPolicy:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rules = yaml.safe_load(self.path.read_text(encoding="utf-8-sig"))
        self._validate()
        self.version = self.rules["version"]

    def _validate(self) -> None:
        rules = self.rules
        if not isinstance(rules, dict) or not all(k in rules for k in
                ("version", "always_block", "context_dependent", "agents")):
            raise ValueError("Неполная YAML-политика")
        if not isinstance(rules["always_block"], list) or not isinstance(rules["agents"], dict):
            raise ValueError("Некорректные always_block/agents")
        known_forbidden = {"disable_audit_or_wipe_logs", "pipe_remote_script_to_shell",
                           "add_ssh_keys_or_sudoers_via_chat", "send_secrets_or_data_to_external_hosts"}
        for agent in rules["agents"].values():
            if not all(k in agent for k in ("roles", "tools", "environments", "forbidden_actions")):
                raise ValueError("Неполная политика агента")
            if set(agent["forbidden_actions"]) - known_forbidden:
                raise ValueError("В политике есть неподдерживаемый forbidden_action")
            for role in agent["roles"].values():
                if role.get("secrets") not in ("deny", "names_only"):
                    raise ValueError("Неподдерживаемое правило secrets")
                for permission in role["tools"].values():
                    if isinstance(permission, str) and permission in ("deny", "any", "nonprod", "allow", "confirm"):
                        continue
                    if (isinstance(permission, dict) and permission
                            and set(permission) <= {"prod", "nonprod", "any"}
                            and all(p in ("allow", "confirm", "deny") for p in permission.values())):
                        continue
                    raise ValueError(f"Некорректное правило инструмента: {permission!r}")

    @staticmethod
    def _result(decision: str, category: str, reason: str, level: str | None = None) -> dict:
        return {"decision": decision, "risk_category": category,
                "risk_level": level or {"allow": "none", "confirm": "medium", "block": "high"}[decision],
                "explanation": reason}

    def environment(self, text: str, agent: dict) -> str:
        found = set()
        if re.search(r"\b(?:prod|production|прод|продакш[ен]\w*)\b", text):
            found.add("prod")
        if re.search(r"\b(?:dev|stage|staging|test|development|тестов\w*)\b", text):
            found.add("nonprod")
        tokens = re.findall(r"[\w-]+", text)
        for environment, patterns in agent["environments"].items():
            if any(fnmatch.fnmatchcase(token, pattern) for token in tokens for pattern in patterns):
                found.add(environment)
        # Для нескольких окружений берём наиболее ограничительное prod.
        return "prod" if "prod" in found else "nonprod" if "nonprod" in found else "unknown"

    @staticmethod
    def tools(text: str) -> list[str]:
        result = []
        patterns = {
            "read_logs": r"\bread_logs\b|(?:покаж|прочита|вывед|посмотр|найд|show|read|get|tail|grep).{0,90}(?:лог|\blogs?\b)",
            "restart_service": r"\brestart_service\b|перезапуст|рестарт|\brestart\b",
            "delete_resource": r"\bdelete_resource\b|(?:удал\w*|\bdelete\b|\bremove\b).{0,80}(?:ресурс|namespace|volume|snapshot|снапшот|бакет|bucket|под\b|\bpod\b)",
            "run_shell": r"\brun_shell\b|выполни|запусти.{0,30}(?:команд|скрипт)|\b(?:run|execute)\b|"
                         r"масштабиру|откати|\b(?:scale|roll back|rollback)\b|"
                         r"(?:очисти|сбрось|flush|clear).{0,25}(?:к[еэ]ш|redis|cache)|"
                         r"\b(?:kubectl|systemctl|chmod|rm -|drop table)\b|(?:удали|очисти).{0,40}таблиц|"
                         r"\bgit\s+(?:push|reset|clean|rebase|commit|restore|rm|cherry-pick|revert|merge|stash|checkout|switch)\b|"
                         r"(?:^|\b(?:сделай|выполни|запусти|run|execute|perform|do)\s+)force[-\s]+push\b",
        }
        for tool, pattern in patterns.items():
            if re.search(pattern, text):
                result.append(tool)
        # Упоминание команды restart/delete через shell требует проверки обоих прав.
        return result

    def evaluate(self, user_prompt: str, context: dict, options: dict) -> dict:
        agent_name, role_name = context["target_agent"], context["user_role"]
        agent = self.rules["agents"].get(agent_name)
        if agent is None or role_name not in agent["roles"]:
            return self._result("block", "tool_abuse", "Неизвестный агент или роль в доверенном контексте.")
        role = agent["roles"][role_name]
        text = action_text(user_prompt, context, options)
        # Объяснение команды не является запросом её выполнить. Классификатор
        # угроз всё равно проверяет этот текст: «учебная цель» не снимает защиту.
        unquoted = re.sub(r"```.*?```|`[^`]*`|«[^»]*»|\"[^\"]*\"", " ", text, flags=re.S)
        explicit_execution = re.search(
            r"(?:^|[.!?;:,\n]\s*|\b(?:и|а затем|а потом|затем|потом|после этого|and(?:\s+then)?|then)\s+)"
            r"(?:(?:пожалуйста|срочно|теперь|сейчас|please|now|first)\s+)?"
            r"(?:выполни|запусти|перезапусти|удали|отключи|очисти|сбрось|масштабируй|откати|"
            r"покажи|выведи|дай|распечатай|прочитай|добавь|пропиши|отправь|перешли|загрузи|"
            r"run|execute|restart|delete|remove|disable|flush|show|print|dump|give|list|read|get|"
            r"add|append|send|forward|upload|perform|do)\b", unquoted)
        if educational(text) and not explicit_execution:
            return self._result("allow", "none", "Информационный запрос; выполнения действия не обнаружено.")

        forbidden_patterns = {
            "disable_audit_or_wipe_logs": r"(?:отключ|останов|disable|stop).{0,35}(?:аудит|audit)|(?:удал|очист|wipe|rm -rf).{0,35}(?:логи|logs|/var/log)",
            "pipe_remote_script_to_shell": r"(?:curl|wget)[^\n]{0,250}\|\s*(?:sudo\s+)?(?:ba)?sh\b",
            "add_ssh_keys_or_sudoers_via_chat": r"(?:добав|пропиш|add|append|echo).{0,100}(?:ssh|authorized_keys|sudoers)",
            "send_secrets_or_data_to_external_hosts": r"(?:отправ|перешли|загрузи|send|forward|upload|post).{0,130}(?:секрет|парол|токен|истори|данны|secret|password|token|user.data).{0,130}(?:https?://|[\w.-]+@[\w.-]+)",
        }
        for name in agent["forbidden_actions"]:
            if re.search(forbidden_patterns[name], text):
                category = "data_exfiltration" if name == "send_secrets_or_data_to_external_hosts" else "tool_abuse"
                return self._result("block", category, f"Политика запрещает действие {name} для любой роли.")

        secret = bool(re.search(r"секрет|парол|токен|ключ[иь].{0,10}api|api.{0,10}ключ|secret|password|token|access.keys|kubeconfig", text))
        secret_request = bool(re.search(r"покаж|вывед|дай|распечат|прочита|show|print|dump|give|list|read|get", text))
        names_only = bool(re.search(r"(?:только|лишь).{0,15}(?:имен|назван)|(?:имена|названия).{0,30}без значен|names? only|only.{0,10}names?|without.{0,10}values", text))
        if secret and secret_request and not (names_only and role["secrets"] == "names_only"):
            return self._result("block", "unauthorized_data_access", f"Роль {role_name}: правило secrets={role['secrets']} запрещает запрошенные секреты.")

        tools = self.tools(text)
        environment = self.environment(text, agent)
        confirmations = []
        for tool in tools:
            permission = role["tools"].get(tool, "deny")
            if tool not in agent["tools"]:
                permission = "deny"
            if isinstance(permission, dict):
                decision = permission.get(environment, permission.get("any", "deny"))
            elif permission in ("any", "allow"):
                decision = "allow"
            elif permission == "nonprod":
                decision = "allow" if environment == "nonprod" else "deny"
            else:
                decision = permission
            if decision == "deny":
                ambiguity = " Окружение неоднозначно; nonprod не предполагается автоматически." if environment == "unknown" else ""
                category = "unauthorized_data_access" if tool == "read_logs" else "tool_abuse"
                return self._result("block", category, f"Роль {role_name} не допускает {tool} в окружении {environment}." + ambiguity)
            if decision == "confirm":
                confirmations.append(tool)
        if confirmations:
            ambiguity = " Окружение не определено; правило действует для любого окружения." if environment == "unknown" else ""
            return self._result("confirm", "sensitive_action", f"Роль {role_name} допускает {', '.join(confirmations)} в {environment} только после отдельного подтверждения." + ambiguity)
        if tools:
            reason = f"Роль {role_name} допускает {', '.join(tools)} в окружении {environment}."
        else:
            reason = "Модель не выявила атаку; запрещённое действие по YAML-политике не обнаружено."
        return self._result("allow", "none", reason)
