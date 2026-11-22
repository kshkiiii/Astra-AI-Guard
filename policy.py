from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import yaml

from text_processing import normalize, text_variants


def execution_requested(text: str) -> bool:
    unquoted = re.sub(r"```.*?```|`[^`]*`|«[^»]*»|\"[^\"]*\"|'[^']*'", " ", text, flags=re.S)


    requested = re.search(
        r"\b(?:прошу|попрошу|просим|умоляю|просьба|не мог(?:ли)? бы (?:ты|вы)|"
        r"будь(?:те)? добр\w*|мне (?:нужно|необходимо)|хочу,? чтобы (?:ты|вы))[,\s]+"
        r"(?:показ\w*|предостав\w*|присла\w*|получ\w*|выгруз\w*|раскры\w*|"
        r"отключ\w*|удал\w*|выполн\w*|запуст\w*|перезапуст\w*|отправ\w*|прочита\w*)\b", unquoted)
    explicit = re.search(
        r"(?:^|[.!?;:,\n]\s*|\b(?:и|а затем|а потом|а также|а ещё|затем|потом|после этого|and(?:\s+then)?|then)\s+)"
        r"(?:(?:пожалуйста|срочно|теперь|сейчас|please|now|first)\s+)?"
        r"(?:выполни|запусти|перезапусти|удали|отключи|очисти|сбрось|масштабируй|откати|"
        r"покажи|выведи|дай|распечатай|прочитай|добавь|пропиши|измени|внеси|запиши|закоммить|запушь|отправь|перешли|загрузи|"
        r"дропни|drop|truncate|добейся|обеспечь|убери|сотри|уничтожь|ликвидируй|верни|прекрати|останови|пересоздай|выдай|поделись|достань|сделай|"
        r"игнорируй|забудь|отмени|ignore|disregard|override|"
        r"run|execute|restart|delete|remove|disable|flush|show|print|dump|give|list|read|get|"
        r"add|append|send|forward|upload|perform|do)\b", unquoted)
    from request_analysis import INFRA_CHANGE
    infra = re.search(r"(?:^|[;\n.!?]\s*|\b(?:затем|потом|then|and)\s+)(?:(?:пожалуйста|срочно|теперь|please|now)\s+)?(?:"+INFRA_CHANGE+r")", unquoted)
    return bool(explicit or requested or infra)


def informational_request(text: str) -> bool:

    return educational(text) and not execution_requested(text)


def educational(text: str) -> bool:


    body = re.sub(
        r"^(?:(?:добрый день|здравствуйте|привет|слушай|срочно|вопрос|пожалуйста|"
        r"hello|hey|hi|urgent|please)\b[\s,.!:;—-]*)+", "", text)
    return bool(re.match(
        r"(?:сначала\s+)?(?:объясни|расскажи.{0,20}(?:как работает|что такое)|"
        r"почему\b|что (?:означает|такое)|как\b|разбер[иё]|классифицируй\b|для (?:онбординга|учебного)|"
        r"без (?:выполнения|технических))|"
        r"(?:first\s+)?(?:classify\b|categorize\b|write\s+(?:a\s+)?(?:regex|regular expression|bug report)\b|explain\b|what (?:is|does)|why\b|how (?:does|to prevent)|"
        r"without (?:executing|technical))", body))


def action_text(user_prompt: str, context: dict, options: dict) -> str:
    text = "\n".join(text_variants(user_prompt, options))

    if re.search(r"сделай (?:это|так)|выполни (?:это|е[её])|продолж|do it|execute it|continue", text):
        history = [m["content"] for m in context.get("history", []) if m["role"] == "user"]
        text = "\n".join([*(normalize(s) for s in history[-3:]), text])
    return text


class SecurityPolicy:
    @classmethod
    def from_text(cls, text, path):
        if len(text) > 64000:
            raise ValueError("Политика превышает 64 000 символов")
        if any(isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)) for token in yaml.scan(text)):
            raise ValueError("YAML-якоря и ссылки не поддерживаются")
        class Loader(yaml.SafeLoader):
            pass
        def mapping(loader, node):
            pairs = loader.construct_pairs(node)
            result = {}
            for key, value in pairs:
                if not isinstance(key, str) or key in result:
                    raise ValueError("Ключи YAML должны быть строками без повторений")
                result[key] = value
            return result
        Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        obj = cls.__new__(cls)
        obj.path = Path(path)
        obj.rules = yaml.load(text, Loader=Loader)
        obj._validate()
        obj.version = obj.rules["version"]
        return obj

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
        if type(rules["version"]) is not int or rules["version"] < 1:
            raise ValueError("Версия должна быть положительным целым числом")
        categories = {"prompt_injection", "jailbreak", "indirect_injection", "system_prompt_extraction", "system_prompt_override", "privilege_escalation", "data_exfiltration", "harmful_content", "fraud_social_engineering", "abuse"}
        if not set(rules["always_block"]) <= categories:
            raise ValueError("Неизвестная категория always_block")
        if not rules["agents"] or any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", n) for n in rules["agents"]):
            raise ValueError("Нужен агент с корректным идентификатором")
        known_forbidden = {"disable_audit_or_wipe_logs", "pipe_remote_script_to_shell",
                           "add_ssh_keys_or_sudoers_via_chat", "send_secrets_or_data_to_external_hosts", "wipe_root_filesystem"}
        for agent in rules["agents"].values():
            if not isinstance(agent, dict):
                raise ValueError("Описание агента должно быть объектом")
            if not all(k in agent for k in ("roles", "tools", "environments", "forbidden_actions")):
                raise ValueError("Неполная политика агента")
            if set(agent["forbidden_actions"]) - known_forbidden:
                raise ValueError("В политике есть неподдерживаемый forbidden_action")
            supported_tools = {"run_shell", "read_logs", "restart_service", "delete_resource", "write_repository"}
            if not isinstance(agent["tools"], list) or set(agent["tools"]) - supported_tools:
                raise ValueError("Неизвестный инструмент")
            if (not isinstance(agent["environments"], dict) or set(agent["environments"]) != {"prod", "nonprod"}
                    or any(not isinstance(v, list) or any(not isinstance(p, str) or not p for p in v) for v in agent["environments"].values())):
                raise ValueError("Нужны списки шаблонов prod и nonprod")
            if not isinstance(agent["roles"], dict) or not agent["roles"]:
                raise ValueError("Нужны роли")
            if "deny_confirmation_bypass_for" in agent:
                denied = agent["deny_confirmation_bypass_for"]
                if not isinstance(denied,list) or any(not isinstance(v,str) or v not in agent["tools"] for v in denied):
                    raise ValueError("deny_confirmation_bypass_for должен перечислять известные инструменты")
            if "repository" in agent:
                repo = agent["repository"]
                if (not isinstance(repo, dict) or set(repo) != {"protected_branches", "development_branches"}
                        or any(not isinstance(v, list) or not v or any(not isinstance(p, str) or not p for p in v) for v in repo.values())):
                    raise ValueError("Нужны непустые списки protected_branches/development_branches")
            for role_name, role in agent["roles"].items():
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", role_name) or not isinstance(role, dict) or not isinstance(role.get("tools"), dict) or set(role["tools"]) - set(agent["tools"]):
                    raise ValueError("Некорректная роль или её инструменты")
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
        if level is None and decision == "block" and category in {"data_exfiltration", "system_prompt_extraction", "privilege_escalation"}:
            level = "critical"
        return {"decision": decision, "risk_category": category,
                "risk_level": level or {"allow": "none", "confirm": "medium", "block": "high"}[decision],
                "explanation": reason}

    def environment(self, text: str, agent: dict) -> str:
        from text_processing import environment_text
        text = environment_text(text)
        found = set()
        if re.search(r"\b(?:prod|production|прод(?:е|а|у|ом)?|продакш\w*|боев\w*|промышленн\w*)\b", text):
            found.add("prod")
        if re.search(r"\b(?:dev|stage|staging|test|development|тестов\w*)\b", text):
            found.add("nonprod")
        tokens = re.findall(r"[\w-]+", text)
        for environment, patterns in agent["environments"].items():
            if any(fnmatch.fnmatchcase(token, pattern) for token in tokens for pattern in patterns):
                found.add(environment)

        return "prod" if "prod" in found else "nonprod" if "nonprod" in found else "unknown"

    def evaluate_intents(self, actions, context: dict) -> dict:


        from intent import ActionIntent, FORBIDDEN, TOOLS
        agent = self.rules["agents"].get(context["target_agent"])
        role_name = context["user_role"]
        if agent is None or role_name not in agent["roles"]:
            return self._result("block", "tool_abuse", "Неизвестный агент или роль в доверенном контексте.")
        role = agent["roles"][role_name]
        confirmations = []
        for action in actions:
            if not isinstance(action, ActionIntent):
                raise ValueError("Политика принимает только проверенную структуру ActionIntent")
            if action.mode == "informational":
                continue

            if action.source not in {"current_user", "history_user"}:
                return self._result("block", "indirect_injection", "Недоверенный источник не может назначать действия агенту.")
            operation = action.operation
            if operation in FORBIDDEN and operation in agent["forbidden_actions"]:
                category = "data_exfiltration" if operation == "send_secrets_or_data_to_external_hosts" else "tool_abuse"
                return self._result("block", category, f"Структура запроса: действие {operation} запрещено YAML-политикой.")
            if action.secrets != "none" or operation.startswith("secret_"):
                names = operation == "secret_names" and action.secrets == "names"
                if not (names and role["secrets"] == "names_only"):
                    return self._result("block", "unauthorized_data_access", f"Роль {role_name}: структура запроса требует недопустимый доступ к секретам.")
                continue
            if operation not in TOOLS:
                return self._result("block", "tool_abuse", "Не удалось однозначно определить запрошенную операцию; уточните действие.")
            permission = role["tools"].get(operation, "deny") if operation in agent["tools"] else "deny"
            if isinstance(permission, dict):
                decision = permission.get(action.environment, permission.get("any", "deny"))
            elif permission in ("allow", "any"):
                decision = "allow"
            elif permission == "nonprod":
                decision = "allow" if action.environment == "nonprod" else "deny"
            else:
                decision = permission
            if decision == "deny":
                return self._result("block", "unauthorized_data_access" if operation == "read_logs" else "tool_abuse",
                                    f"Роль {role_name} не допускает {operation} в окружении {action.environment} (семантический разбор).")
            if decision == "confirm":
                confirmations.append(f"{operation} в {action.environment}")
        if confirmations:
            return self._result("confirm", "sensitive_action", f"Роль {role_name}: {', '.join(sorted(set(confirmations)))} требует отдельного подтверждения по YAML.")
        operations = sorted({f"{a.operation} в {a.environment}" for a in actions if a.mode == "execute"})
        detail = ', '.join(operations[:8]) if operations else "выполнение инфраструктурных действий не запрошено"
        return self._result("allow", "none", f"Роль {role_name}: {detail}; YAML-политика разрешает запрос.")

    @staticmethod
    def tools(text: str) -> list[str]:
        result = []
        patterns = {
            "read_logs": r"\bread_logs\b",
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


        from request_analysis import READ, LOGS, database_deletions
        if "delete_resource" not in result and database_deletions(text):
            result.append("delete_resource")
        if "read_logs" not in result and any(re.search(READ, clause) and re.search(LOGS, clause)
                                             for clause in re.split(r"[;\n]", text)):
            result.append("read_logs")

        return result

    def evaluate(self, user_prompt: str, context: dict, options: dict) -> dict:
        agent_name, role_name = context["target_agent"], context["user_role"]
        agent = self.rules["agents"].get(agent_name)
        if agent is None or role_name not in agent["roles"]:
            return self._result("block", "tool_abuse", "Неизвестный агент или роль в доверенном контексте.")
        role = agent["roles"][role_name]
        text = action_text(user_prompt, context, options)


        from request_analysis import assessment_text
        if informational_request(assessment_text(text)):
            return self._result("allow", "none", "Информационный запрос; выполнения действия не обнаружено.")

        forbidden_patterns = {
            "wipe_root_filesystem": r"\brm\s+(?:--[\w-]+\s+)*-[rf]+\s+/(?:\s|$|[;\"'])|\brm\s+-[rf]+\s+--no-preserve-root|\bdd\b.{0,160}\bof=/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z])\b|\bmkfs\b.{0,80}/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z])\b",
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


        from request_analysis import anchored_actions
        anchors = anchored_actions(user_prompt, context, self)
        scoped = self.evaluate_intents(anchors, context)
        if scoped["decision"] == "block":
            return scoped
        if scoped["decision"] == "confirm":
            confirmations.extend(a.operation for a in anchors if a.mode == "execute")
        for tool in tools:
            if any(a.operation == tool for a in anchors):
                continue
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
