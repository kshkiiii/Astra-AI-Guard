
from __future__ import annotations

import re
import time

from intent import ActionIntent, SemanticSecurityModel
from policy import execution_requested, informational_request
from text_processing import normalize, prepare_input, text_variants
from request_analysis import anchored_actions, assessment_text


def strongest(results):
    return max(results, key=lambda r: {"allow": 0, "confirm": 1, "block": 2}[r["decision"]])


def windows(text: str, size=1400, overlap=200):


    for start in range(0, max(1, len(text)), size - overlap):
        yield text[start:start + size]
        if start + size >= len(text):
            break


def resolve_reference(prompt, context):
    reference = re.fullmatch(
        r"[\s,.!]*(?:(?:да|ок|okay|yes)[,\s]*)?(?:сделай(?: это| так)?|выполни(?: это| её| ее)?|"
        r"продолжай|продолжи|да|давай|действуй|согласен|подтверждаю|впер[её]д|do it|execute it|continue|yes|proceed|go ahead)[\s,.!]*", normalize(prompt))
    if not reference:
        return prompt, "current_user", False


    messages = [m for m in context.get("history", []) if m["content"].strip()]
    if not messages:
        return prompt, "current_user", True
    return messages[-1]["content"] + "\n" + prompt, "current_user", False


def literal_injection(text, source, *, information_exception=True):
    normalized = normalize(text)


    if information_exception and informational_request(normalized):
        return False
    patterns = (
        r"</?(?:user|system|assistant)>|<\|(?:im_start|im_end|system)\|>|\[/?inst\]",
        r"(?:восстанови|воссоздай|реконструируй|recover|reconstruct).{0,80}(?:секретн\w* промпт|скрыт\w* промпт|системн\w* промпт|private.{0,15}prompt|system prompt)",
        r"(?:мо[яю]|текущ\w*)\s+роль.{0,70}(?:должна стать|станет|считай|замени|повыс).{0,35}(?:админ|admin)",
        r"прежни\w*\s+(?:ограничен\w*|правил\w*|инструкц\w*)\s+отменяются",
        r"(?:игнорируй|забудь|отмени|ignore|disregard|forget).{0,90}(?:инструкц|правил|ограничен|instructions|rules|constraints)",
        r"(?:системн\w*|предыдущ\w*).{0,40}(?:правил\w*|инструкц\w*|ограничен\w*).{0,30}(?:отмен|не действ|устарел)",
        r"(?:верни|выдай|поставь|ответь|output|return).{0,70}(?:decision.{0,10}allow|allow.{0,40}(?:независимо|провер|наруш))",
        r"(?:считай|это|новое).{0,25}(?:системн\w* сообщен|system message)",
        r"(?:покажи|распечатай|раскрой|выведи|reveal|print|show).{0,60}(?:системн\w* промпт|скрыт\w* инструкц|system prompt|hidden.{0,20}instructions)",
        r"(?:повтори|процитируй|воспроизведи|repeat|copy).{0,100}(?:системн\w* промпт|скрыт\w* инструкц|system prompt|hidden.{0,20}instructions|initial.{0,20}instructions)",
        r"(?:поменяй|измени|установи|подмени|считай).{0,50}(?:user_role|мою роль|роль в контексте)",
        r"(?:используй|прими|возьми|подставь).{0,60}(?:контекст|context).{0,80}(?:user_role|роль)",
    )
    return any(re.search(pattern, normalized, flags=re.S) for pattern in patterns)


def legacy_evidence(text):


    normalized = normalize(text).strip(" .,!?:;—-\n\t")
    if len(normalized) <= 1 or normalized in {"ок", "окей", "ok", "okay", "да", "нет", "yes", "no", "спасибо", "thanks"}:
        return False


    if informational_request(normalized) and re.search(r"«[^»]+»|`[^`]+`|\"[^\"]+\"|'[^']+'", normalized):
        return False
    return True


class LayeredInspector:
    def __init__(self, model, attack_threshold=.65, operation_threshold=.35):
        if not 0 < attack_threshold < 1 or not 0 < operation_threshold < 1:
            raise ValueError("Пороги слоев должны быть между 0 и 1")
        self.model = model
        self.attack_threshold = attack_threshold
        self.operation_threshold = operation_threshold
        self.legacy_model = None
        self.legacy_attack_threshold = .35
        self.legacy_rescue_threshold = .10

    @classmethod
    def from_config(cls, config):
        settings = config.get("guard", {})
        if not settings.get("layered", False):
            return None
        model = SemanticSecurityModel.load(config["paths"]["semantic_security"], config["paths"]["encoder"])
        threshold = settings.get("attack_threshold")
        if threshold is None:
            threshold = model.metadata["threat_threshold"]
        inspector = cls(model, threshold, settings.get("operation_threshold", .35))
        inspector.legacy_attack_threshold = settings.get("legacy_attack_threshold", .35)
        inspector.legacy_rescue_threshold = settings.get("legacy_rescue_threshold", .10)
        if not 0 < inspector.legacy_attack_threshold < 1 or not 0 < inspector.legacy_rescue_threshold < 1:
            raise ValueError("Некорректный порог независимой модели")
        return inspector

    def check(self, prompt, context, options, policy, diagnostics=False):
        started = time.perf_counter()
        prepared = prepare_input(prompt, context, options)
        resolved, source, unresolved = resolve_reference(prompt, context)
        views = []
        literal_views = []
        for segment in prepared.segments:
            for variant in text_variants(segment.text, options):
                literal_views.extend((segment.source, text) for text in windows(variant) if text.strip())
                cleaned = assessment_text(variant)
                views.extend((segment.source, text) for text in windows(cleaned) if text.strip())
        if resolved != prompt:
            views.extend((source, text) for text in windows(normalize(resolved)))
        views = list(dict.fromkeys(views))
        if not views:
            views = [("current_user", "")]

        predictions = self.model.predict([text for _, text in views])
        attack_index = max(range(len(views)), key=lambda i: predictions[i]["attack_probability"])
        attack = predictions[attack_index]
        attack_source = views[attack_index][0]
        neural = policy._result("allow", "none", "Нейросетевой детектор не обнаружил атаку.")
        if attack["attack_probability"] >= self.attack_threshold:
            category = "indirect_injection" if attack_source == "attachment" else attack["risk_category"]
            neural = policy._result("block", category, f"Нейросетевой детектор: {category}, источник {attack_source}.")
        literal = policy._result("allow", "none", "Проверка исходного текста пройдена.")
        for view_source, text in literal_views:


            directive = (view_source == "attachment" and re.search(r"для (?:бота|ассистента)|(?:run_shell|restart_service|delete_resource)\s*\(", text))
            if literal_injection(text, view_source) or (directive and not informational_request(text)):
                category = "indirect_injection" if view_source == "attachment" else "prompt_injection"
                literal = policy._result("block", category, f"В исходном тексте обнаружена попытка подменить инструкции; источник {view_source}.")
                break
        raw = policy.evaluate(resolved, context, options)
        actions = self.extract_actions(resolved, context, policy, options)
        structured = policy.evaluate_intents(actions, context)
        if unresolved:
            structured = policy._result("block", "tool_abuse", "Для продолжения не найдено действие в истории; уточните запрос.")
        legacy = policy._result("allow", "none", "Дополнительная модель не обнаружила атаку.")
        legacy_score = 0.
        if self.legacy_model is not None:
            candidates = [(s, t) for (s, t), p in zip(views, predictions)
                          if p["attack_probability"] >= self.legacy_rescue_threshold
                          and legacy_evidence(t)
                          and not (informational_request(t) and p["execute_probability"] < .10)]
            if candidates:


                candidate_set = set(candidates)
                segments = [(s.source, assessment_text(s.text)) for s in prepared.segments]
                if resolved != prompt:
                    segments.append((source, resolved))
                legacy_views = []
                for segment_source, body in segments:
                    variants = text_variants(body, options)
                    if not any((segment_source, part) in candidate_set for variant in variants for part in windows(variant)):
                        continue
                    joined = "\n[variant]\n".join(variants)
                    legacy_views.extend(f"[{segment_source}]\n{part}" for part in windows(
                        joined, options.get("chunk_size", 3000), options.get("chunk_overlap", 300)))
                observation = self.legacy_model.analyze(tuple(dict.fromkeys(legacy_views)))
                legacy_score = observation["attack_probability"]
                if legacy_score >= self.legacy_attack_threshold and observation["risk_category"] in policy.rules["always_block"]:
                    legacy = policy._result("block", observation["risk_category"],
                                            "Дополнительная модель обнаружила угрозу; семантический детектор не подтвердил безопасность фрагмента.")
        checks = [("literal_injection", literal), ("neural_threat", neural), ("legacy_threat", legacy), ("text_policy", raw), ("intent_policy", structured)]
        result = dict(strongest([structured, raw, literal, neural, legacy]))
        result.update(attack_probability=attack["attack_probability"], policy_version=policy.version,
                      latency_ms=round((time.perf_counter() - started) * 1000, 3),
                      intent=[a.to_dict() for a in actions],
                      checks=[{"layer": name, "decision": r["decision"], "risk_category": r["risk_category"]} for name, r in checks])
        if diagnostics:
            result["diagnostics"] = {"attack_source": attack_source, "views": len(views),
                                     "attack_threshold": self.attack_threshold, "legacy_score": legacy_score,
                                     "resolved_reference": resolved != prompt}
        return result

    def extract_actions(self, text, context, policy, options):
        anchors = anchored_actions(text, context, policy)
        text = assessment_text(text)


        clauses = [text, *re.split(r"[;\n]|(?<=[.!?])\s+|\b(?:а затем|а потом|затем|потом|and then|then|и|and)\s+", text)]
        clauses = list(dict.fromkeys(c.strip() for c in clauses if c.strip()))
        if not clauses:
            return anchors
        predictions = self.model.predict(clauses)
        agent = policy.rules["agents"].get(context["target_agent"])
        if agent is None:
            return []
        informational = informational_request(text)
        actions = list(anchors)
        for index, (clause, prediction) in enumerate(zip(clauses, predictions)):
            operation = prediction["operation"]


            if anchors and operation in {"none", "read_logs", "write_repository"}:
                continue
            mode = "informational" if informational else "execute" if execution_requested(clause) or prediction["execute_probability"] >= .5 else "informational"


            if operation == "none" and not re.search(r"(?:добейся|обеспечь|пусть|сделай так).{0,100}(?:сервис|ресурс|запуск|процесс|аудит|парол|токен)", clause):
                continue
            if prediction["confidence"] < self.operation_threshold:
                if mode == "informational":
                    continue


                if policy.tools(clause):
                    continue
                if not re.search(r"добейся|обеспечь|пусть.{0,80}(?:сервис|ресурс)|сделай так.{0,80}(?:сервис|ресурс)", clause):
                    continue
                operation = "none"


            if operation == "secret_names" and not re.search(
                    r"(?:только|лишь).{0,15}(?:имен|назван)|(?:имен|назван).{0,40}(?:без значен|значения скрой|не раскрывая)|"
                    r"names? only|only.{0,10}names?|without.{0,10}values", clause):
                operation = "secret_values"

            if index and mode == "informational":
                continue
            environment = policy.environment(clause, agent)
            if environment == "unknown":


                prod = bool(re.search(r"\b(?:prod|production|прод|боев\w*)\b|\w+-prod(?:-|\b)", text))
                nonprod = bool(re.search(r"\b(?:dev|stage|staging|test|тестов\w*)\b|\b(?:dev|stage|test)-\w+", text))
                if prod != nonprod:
                    environment = "prod" if prod else "nonprod"
            objects = re.findall(r"\b[\w]+(?:[-_][\w]+)+\b", clause)
            target = next((o for o in objects if o not in {"read_logs", "run_shell", "restart_service", "delete_resource"}), None)
            if target is None:
                match = re.search(r"\b(?:сервис[ау]?|служб[ау]|ресурс[ау]?|namespace|volume|bucket|бакет[ау]?|секретов)\s+([\w.-]+)", clause)
                if match and match[1] not in {"в", "на", "и", "для", "prod", "dev", "production", "stage"}:
                    target = match[1]
            action = ActionIntent(operation=operation, object=target, environment=environment,
                                  secrets="values" if operation == "secret_values" else "names" if operation == "secret_names" else "none",
                                  mode=mode, source="current_user", confidence=prediction["confidence"])
            same = next((i for i, a in enumerate(actions) if
                         (a.operation, a.object, a.environment, a.source) ==
                         (action.operation, action.object, action.environment, action.source)), None)
            if same is None:
                actions.append(action)
            elif (action.mode == "execute", action.confidence) > (actions[same].mode == "execute", actions[same].confidence):
                actions[same] = action
        return actions
