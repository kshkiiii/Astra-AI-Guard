from __future__ import annotations

import fnmatch
import re

from intent import ActionIntent
from policy import execution_requested, informational_request
from text_processing import normalize

READ = r"\b(?:покаж\w*|показ\w*|прочита\w*|вывед\w*|посмотр\w*|найд\w*|дай|дайте|достань|пришли|пришлите|присла\w*|получи\w*|получить|предостав\w*|выгруз\w*|подели\w*|ознаком\w*|передай|вытащи|show|read|get|give|fetch|retrieve|send|tail|grep|use)\b"
LOGS = r"\b(?:лог\w*|журнал\w*|logs?|logfile)\b"
MUTATE = r"\b(?:дропни|дропнуть|drop|добав\w*|внеси|внести|измени\w*|изменить|исправ\w*|впиши|запиши|записать|закоммит\w*|залей|запуш\w*|закинь|закинуть|удал\w*|убери|отредактир\w*|обнови|влей|вмерж\w*|merge|commit|push|edit|modify|change|add|delete|remove|update)\b"
REPOSITORY = r"\b(?:ветк\w*|веточк\w*|branch|репозитори\w*|repository|repo)\b|(?:prod|production)[- ]ветк"
INFRA_CHANGE = r"\b(?:deploy|redeploy|rollout|configure|д[еэ]плой\w*|задеплой\w*|разверни|развернуть|перевыкати|выкати|выкатить)\b|\b(?:поставь|установи|измени|смени|выставь|set|change|update)\b.{0,100}(?:\b[a-z_][a-z_0-9]*\s*=|log_level|настройк|конфигурац|configuration|replicas|реплик)"



DATABASE_DELETE = re.compile(
    r"(?:^|[;\n.!?:]\s*|\b(?:затем|потом|then|and)\s+)\s*"
    r"(?:(?:пожалуйста|срочно|теперь|сейчас|please|now|прошу)\s+)?"
    r"(?:удали|сотри|уничтожь|дропни|drop|delete|remove|truncate)\s+"
    r"(?:(?:полностью|тестовую|the|test)\s+)?"
    r"(?:баз[ауые](?:\s+данных)?|бд|database|schema|таблиц[ауые]|table)\b"
    r"(?:\s+if\s+exists)?\s*(?:[`\"\[](?P<quoted>[\w.$-]+)[`\"\]]|(?P<plain>[\w.$-]+))?"
)


def database_deletions(text):
    return [match["quoted"] or match["plain"] for match in DATABASE_DELETE.finditer(text)]


def assessment_text(text):
    original = normalize(text)
    text = original
    text = re.sub(r"\bя\s+(?:админ(?:истратор)?|admin(?:istrator)?|sre|developer|разработчик)\b", " ", text)
    text = re.sub(r"\b(?:i am|i'm)\s+(?:an?\s+)?(?:admin(?:istrator)?|developer|sre)\b", " ", text)
    text = re.sub(r"\bс другого (?:акка|аккаунта) заш[её]л\b", " ", text)
    text = re.sub(r"^(?:вы зашли|ты заш[её]л) с другого аккаунта[.!\s]*$", " ", text)
    text = re.sub(r"(?<!\w)([a-zа-яё])\1{11,}(?!\w)", " ", text)


    if text == original:
        return original
    if not re.search(r"\w", text):
        return ""
    return re.sub(r"[ \t]+", " ", text).lstrip(" ,;:.!\n\t").rstrip(" ,;:\n\t")


def discussion(text):
    return informational_request(text)


def branch_target(text):
    after = re.search(r"\b(?:ветк\w*|branch)\s+[`\"']?([\w./-]+)", text)
    before = re.search(r"\b(prod(?:uction)?|прод|боев\w*)[- ]+(?:ветк\w*|branch)\b", text)
    if before:
        return "prod" if before[1].startswith(("прод", "боев")) else before[1]
    return after[1] if after else None


def branch_environment(branch, agent):
    settings = agent.get("repository", {})
    if branch and any(fnmatch.fnmatchcase(branch, p) for p in settings.get("protected_branches", [])):
        return "prod"
    if branch and any(fnmatch.fnmatchcase(branch, p) for p in settings.get("development_branches", [])):
        return "nonprod"
    return "unknown"


def anchored_actions(text, context, policy):


    text = assessment_text(text)
    agent = policy.rules["agents"].get(context["target_agent"])
    if not text or agent is None:
        return []
    informational = discussion(text)
    clauses = re.split(r"[;\n]|(?<=[.!?])\s+|\b(?:а затем|а потом|затем|потом|and then|then|и|and)\s+", text)
    actions = []
    previous_read = False
    for clause in dict.fromkeys(clauses):
        mode = "informational" if informational else "execute"
        env = policy.environment(clause, agent)
        if env == "unknown":


            environments = {policy.environment(c, agent) for c in clauses} - {"unknown"}
            if len(environments) == 1:
                env = environments.pop()
        objects = re.findall(r"\b\w+(?:[-_]\w+)+\b", clause)
        obj = objects[0] if objects else None
        operations = []
        from request_framing import framing
        for database in database_deletions(clause):
            actions.append(ActionIntent("delete_resource", database, env, "none", mode, "current_user", 1.))
        if re.search(INFRA_CHANGE, framing(clause)):
            operations.append("run_shell")


        data_demand = re.match(r"\s*(?:пожалуйста[,\s]+)?(?:мне\s+)?(?:нуж\w*|необходим\w*|требу\w*|дай|покаж\w*|вывед\w*|пришли|предостав\w*|верни)\b", clause)
        secret_subject = re.search(r"секрет|парол|токен|credential|password|secret", clause)
        if data_demand and secret_subject and not informational_request(clause):
            names_only = bool(re.search(r"(?:имен|назван|names?).{0,80}(?:без (?:значен|содержим)|не раскрыва|without.{0,10}values)|(?:только|only).{0,20}(?:имен|назван|names?)", clause))
            if names_only or re.search(r"значени|содержим|парол|токен|credential|password|values|contents", clause):
                actions.append(ActionIntent("secret_names" if names_only else "secret_values", obj, env,
                    "names" if names_only else "values", "execute", "current_user", 1.))
        demand = re.search(READ, clause) or re.search(r"\b(?:мне нужны|нужны|необходимы|требуются|нужен доступ|хочу получить|i need)\b", clause)
        nominal = re.match(r"\s*(?:логи|журнал|logs?)\b", clause)
        if re.search(LOGS, clause) and (demand or nominal or previous_read):
            operations.append("read_logs")
            previous_read = True
        elif execution_requested(clause):
            previous_read = False
        if re.search(MUTATE, clause) and re.search(REPOSITORY, clause):
            branch = branch_target(clause)
            actions.append(ActionIntent("write_repository", branch, branch_environment(branch, agent), "none", mode, "current_user", 1.))
        if re.search(r"\b(?:неизвестн\w*|неуказан\w*|непонятн\w*|unknown|unspecified)\b", clause) and re.search(r"операци|operation|действи", clause) and execution_requested(clause):
            operations.append("none")
        for operation in operations:
            actions.append(ActionIntent(operation, obj, env, "none", mode, "current_user", 1.))

    return list(dict.fromkeys(actions))
