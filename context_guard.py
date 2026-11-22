
import re
import time
from dataclasses import replace

from context_model import ContextModel, request_frame
from content_moderator import ContentModerator
from intent import ActionIntent
from layers import LayeredInspector, literal_injection, resolve_reference, strongest, windows
from model import GuardModel
from policy import SecurityPolicy, execution_requested
from request_analysis import LOGS, anchored_actions, assessment_text
from request_framing import QUOTES, framing
from text_processing import decoded_payloads, prepare_input, text_variants


class ContextGuard:
    backend = "context-minilm-qwen3guard-cpu"
    deadline_seconds = 2.8

    def __init__(self, config, policy=None):
        self.config = config
        self.model = GuardModel.load(config["paths"]["model"], encoder=config["paths"]["encoder"])
        self.baseline_model = GuardModel.load(config["paths"].get("baseline_model", config["paths"]["model"]), encoder=config["paths"]["encoder"])
        self.policy = policy or SecurityPolicy(config["paths"]["policy"])
        self.inspector = LayeredInspector.from_config(config)
        self.inspector.legacy_model = self.model
        self.context_model = ContextModel(config["paths"]["context_model"])
        self.moderator = ContentModerator(config["moderator"])

    def is_ready(self):
        return self.moderator.client.ready()

    def content_check(self, text, contextual_review=False):
        variants = text_variants(text,self.config["text"])


        verdict = dict(safety="Safe",categories=["None"])
        for variant in variants:
            current_verdict = self.moderator.assess(variant)
            if current_verdict["safety"] == "Unsafe":
                verdict = current_verdict
                break
            if current_verdict["safety"] == "Controversial":
                verdict = current_verdict
        allowed = verdict["safety"] == "Safe" or (verdict["safety"] == "Controversial" and contextual_review)
        result = self.policy._result("allow" if allowed else "block", "none" if allowed else "harmful_content",
            "Специализированная проверка содержимого пройдена." if allowed else "В запросе обнаружены инструкции, которые могут причинить вред или раскрыть защищённые данные.")
        result["checks"] = [dict(layer="qwen_content",decision=result["decision"],risk_category=result["risk_category"],safety=verdict["safety"],categories=verdict["categories"])]
        return result

    def check(self, prompt, context, diagnostics=False):
        from inference_budget import inference_budget
        try:
            with inference_budget(self.deadline_seconds-.15):
                return self._check(prompt, context, diagnostics)
        finally:
            self.model.clear_cache()
            self.baseline_model.clear_cache()
            self.inspector.model.clear_cache()

    def _check(self, prompt, context, diagnostics):
        started = time.perf_counter()
        prepare_input(prompt, context, self.config["text"])
        agent = self.policy.rules["agents"].get(context["target_agent"])
        def finish(result, observation=None):
            result.update(backend=self.backend, policy_version=self.policy.version,
                latency_ms=round((time.perf_counter()-started)*1000, 3))
            result.setdefault("intent", [])
            result.setdefault("attack_probability", 0.)
            result.setdefault("checks", [])
            if diagnostics:
                result.setdefault("diagnostics", {})["contextual_semantics"] = observation
            return result
        if agent is None or context["user_role"] not in agent["roles"]:
            return finish(self.policy._result("block", "tool_abuse", "Неизвестная доверенная роль или агент."))
        resolved, _, missing = resolve_reference(prompt, context)
        if missing:
            return finish(self.policy._result("block", "tool_abuse", "Не найдено действие для продолжения; уточните запрос."))
        current = assessment_text(resolved)
        outer = framing(current)


        frame = request_frame(resolved)
        observation = self.context_model.predict([frame])[0]
        safe_probability = observation["safe_probability"]
        safe_threshold = self.context_model.metadata.get("scope_threshold", self.context_model.metadata.get("safe_threshold", .90))
        review_evidence = bool(re.search(r"инъекц|атак|безопаснос|учебн|обучени|перехвачен|security|attack|injection|training|case study|документац|readme|policy", outer))
        review_evidence = review_evidence or bool(re.search(r"фраз|строк|инструкци|цитат|phrase|instruction|quotation", outer) and re.search(r"почему|опасн|смысл|означ|анализ|разбор|why|danger|meaning|analy", outer))
        review = safe_probability >= safe_threshold and review_evidence
        information = safe_probability >= safe_threshold and not review
        clauses = re.split(r"[;\n]|(?<=[.!?])\s+|\b(?:затем|потом|then|и|and)\s+", outer)
        active = literal_injection(outer, "current_user", information_exception=False)
        for clause in clauses:
            content_only = re.search(r"\b(?:дай|верни|выведи|покажи|предоставь|give|return|show)\s+(?:кратк\w*\s+|подробн\w*\s+|только\s+)?(?:оценк|отч[её]т|статус|пример|план|пояснен|объяснен|summary|report|example|explanation)", clause)
            if execution_requested(clause) and not content_only:
                active = True


        plan_task = bool(re.search(r"(?:подтверди|проверь|оцени|сверь|review|assess|confirm|check).{0,25}(?:план|plan)", outer))
        rewrite_task = bool(re.search(r"перепиши|переформулируй|rewrite|rephrase", outer) and re.search(r"документац|readme|описани|раздел|стиль|documentation|description|style", outer))
        report_task = bool(re.search(r"отч[её]т.{0,40}(?:статус|доступност)|status report|health report", outer))
        evidence_review = bool(re.search(r"(?:разбер\w*|проанализ\w*|анализ|разбор|review|analy[sz])", outer) and re.search(r"(?:перехвачен|образец|пример|кейс|sample|captured|case study)", outer) and review_evidence)
        structured_scope = (plan_task or rewrite_task or report_task or evidence_review) and not active
        if structured_scope:
            review = review_evidence
            information = not review
        outer_anchors = [] if structured_scope else anchored_actions(current if active or observation["execute"] >= .5 else outer, context, self.policy)
        originals = anchored_actions(current, context, self.policy)
        for operation in {a.operation for a in outer_anchors}:
            indices = [i for i,a in enumerate(outer_anchors) if a.operation == operation]
            targets = [a for a in originals if a.operation == operation]
            if len(indices) == len(targets):
                for i, target in zip(indices, targets):
                    if outer_anchors[i].object is None and target.object is not None:
                        outer_anchors[i] = replace(outer_anchors[i], object=target.object, environment=target.environment)


        if not structured_scope and observation["execute"] >= .5 and not any(a.operation == "read_logs" for a in outer_anchors):
            for clause in re.split(r"[;\n]|(?<=[.!?])\s+|\b(?:затем|потом|then)\s+", outer):
                if re.search(LOGS, clause):
                    outer_anchors.append(ActionIntent("read_logs", None, self.policy.environment(clause, agent), "none", "execute", "current_user", observation["execute"]))
        outer_permission = self.policy.evaluate_intents(outer_anchors, context)
        if outer_anchors and outer_permission["decision"] == "block":
            return finish(dict(outer_permission, intent=[a.to_dict() for a in outer_anchors],
                checks=[dict(layer="yaml_permissions", decision="block", risk_category=outer_permission["risk_category"])]), observation)


        safe_scope = (review or information) and not active and not any(a.mode == "execute" for a in outer_anchors)


        payloads = [m.group() for m in QUOTES.finditer(resolved)] + [a["content"] for a in context.get("attachments", [])]
        if self.config["text"].get("decode_base64",True):
            payloads.extend(decoded_payloads(resolved))
        payloads += re.findall(r"(?m)^\s*(?:user|assistant|system|ai):.*$|^.*\[system update\].*$", current)
        source_attack = False
        for text in payloads:
            for variant in text_variants(text, self.config["text"]):
                if literal_injection(variant, "attachment", information_exception=False) or "[system update]" in variant or re.search(r"(?:\bai\b|assistant|для (?:бота|ассистента)).{0,100}(?:call.{0,20}tool|выполни|run_shell|delete_database)", variant):
                    source_attack = True
                elif any(p["attack_probability"] >= self.inspector.attack_threshold for p in self.inspector.model.predict(list(windows(variant)))):
                    source_attack = True
        history_attack = any(literal_injection(v, "history_user", information_exception=False) or "[system update]" in v
            for msg in context.get("history", []) for v in text_variants(msg["content"], self.config["text"]))
        if history_attack or (source_attack and not (safe_scope and review)):
            return finish(dict(self.policy._result("block", "indirect_injection", "В недоверенном источнике найдены активные инструкции без подтверждённого разбора атаки."),
                attack_probability=1., checks=[dict(layer="source_boundary", decision="block", risk_category="indirect_injection")]), observation)
        content_text = outer if review or rewrite_task else current
        contextual_review = review_evidence or structured_scope or (safe_scope and bool(re.search(r"реагирован|восстановлен|ротаци|после утечки|respond|recover|rotat",outer)))
        moderation_cache = []
        def moderate():
            if not moderation_cache:
                moderation_cache.append(self.content_check(outer if review or rewrite_task else resolved,contextual_review))
            return moderation_cache[0]
        impersonation = re.search(r"от имени|impersonat|pretend to be",outer) and re.search(r"cto|ceo|директор|админ",outer) and re.search(r"токен|парол|secret|password|credential",outer)
        if impersonation and not review:
            return finish(dict(self.policy._result("block","fraud_social_engineering","Запрошена выдача себя за доверенное лицо для получения защищённых данных."),attack_probability=1.),observation)


        if safe_scope:


            network = re.search(r"\bcurl\b|\bwget\b|https?://|внешн\w* (?:ip|адрес|сервер)", content_text)
            secret_data = re.search(r"/etc/passwd|env_vars|переменн\w* окружени|значени\w* секрет|парол|credentials|password", content_text)
            if network and secret_data:
                return finish(dict(self.policy._result("block", "data_exfiltration", "Запрошены конкретные инструкции передачи защищённых данных во внешнюю сеть."), attack_probability=1.), observation)
            legacy = self.baseline_model.analyze(("[current_user]\n"+outer,))
            if legacy["attack_probability"] >= self.config["model"]["attack_threshold"]:
                moderation = moderate()
                if moderation["decision"] == "block":
                    return finish(dict(moderation,attack_probability=legacy["attack_probability"]),observation)
            return finish(dict(self.policy._result("allow", "none", "Контекстная модель распознала обсуждение без выполнения; активных инструкций вне разбираемых данных не найдено."),
                diagnostics=dict(scope_accepted=True),
                checks=[dict(layer="contextual_semantics", decision="allow", risk_category="none"), dict(layer="yaml_permissions", decision="allow", risk_category="none")]), observation)


        result = self.inspector.check(resolved, context, self.config["text"], self.policy, diagnostics)
        if diagnostics:
            result.setdefault("diagnostics",{})["resolved_reference"] = resolved != prompt
        anchors = outer_anchors or anchored_actions(resolved, context, self.policy)
        if anchors:
            permission = self.policy.evaluate_intents(anchors, context)
            if permission["decision"] == "block":
                result.update(permission)
        raw = self.policy.evaluate(resolved, context, self.config["text"])
        literal = literal_injection(current,"current_user",information_exception=False)
        destructive_bypass = bool(set(self.policy.tools(current)) & set(agent.get("deny_confirmation_bypass_for",[]))) and bool(re.search(r"без (?:подтверждени|согласовани)|without (?:confirmation|approval)",current))
        if destructive_bypass:
            return finish(dict(self.policy._result("block","tool_abuse","Для разрушительной операции запрещён обход отдельного подтверждения."),attack_probability=1.),observation)
        denied_intent = observation["execute"] >= .5 and any(c["layer"] == "intent_policy" and c["decision"] == "block" for c in result["checks"])
        neural_blocks = any(c["layer"] == "neural_threat" and c["decision"] == "block" for c in result["checks"])
        content_category = result["risk_category"] in {"harmful_content","fraud_social_engineering","abuse"}
        if raw["decision"] != "block" and not literal and not denied_intent and (not neural_blocks or content_category):
            if result["decision"] == "block":
                moderation = moderate()
                if moderation["decision"] == "allow":
                    permission = self.policy.evaluate_intents(anchors,context)
                    result.update(strongest([raw,permission]))
                    result["intent"] = [a.to_dict() for a in anchors]
                    result["checks"].append(dict(layer="qwen_content",decision="allow",risk_category="none"))
        result["checks"].append(dict(layer="contextual_semantics", decision=result["decision"], risk_category=result["risk_category"]))
        if result["decision"] == "allow" and not anchors:
            original_view = prepare_input(resolved, dict(context,history=[],attachments=[]), self.config["text"])
            baseline = self.baseline_model.analyze(original_view.chunks)
            if baseline["attack_probability"] >= self.config["model"]["attack_threshold"]:
                protected_execution = observation["execute"] >= .5 and bool(re.search(r"prod|pr0d|production|прод|namespace|сервис|туннел|tunnel|ngrok|ssh|sudo|/etc/|секрет|парол|credentials",current))
                moderation = self.policy._result("block",baseline["risk_category"],"Не удалось надёжно определить и проверить права на запрошенную операцию; уточните действие.") if protected_execution else moderate()
                if moderation["decision"] == "block":
                    result.update(moderation)
                    result["attack_probability"] = baseline["attack_probability"]
                    result["checks"].append(dict(layer="baseline_threat",decision="block",risk_category=baseline["risk_category"]))
        return finish(result, observation)
