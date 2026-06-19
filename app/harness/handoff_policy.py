import re

from app.state.conversation_state import CustomerServiceState


class HandoffPolicy:
    high_risk_terms = [
        "lawyer",
        "legal action",
        "legal complaint",
        "lawsuit",
        "fraud",
        "scam",
        "angry",
        "complaint",
        "unsafe",
        "injury",
        "injuries",
        "injured",
        "large compensation",
        "high compensation",
        "substantial compensation",
        "不安全",
        "受伤",
        "高额赔偿",
        "巨额赔偿",
    ]
    human_patterns = [
        re.compile(r"\bhuman\s+(?:agent|representative|support)\b"),
        re.compile(
            r"\b(?:i\s+)?(?:need|want)\s+(?:a|an)\s+"
            r"(?:human|agent|representative)\b"
        ),
        re.compile(
            r"\b(?:speak|talk)\s+(?:to|with)\s+(?:(?:a|an)\s+)?"
            r"(?:human|agent|representative)\b"
        ),
        re.compile(
            r"\bconnect\s+me\s+(?:to|with)\s+(?:a\s+)?"
            r"(?:human|agent|representative)\b"
        ),
        re.compile(
            r"\btransfer\s+me\s+to\s+(?:(?:a|an)\s+)?"
            r"(?:human|agent|representative)\b"
        ),
        re.compile(r"\b(?:live|real)\s+(?:agent|representative|person)\b"),
        re.compile(r"\bcustomer\s+service\s+(?:agent|representative)\b"),
    ]
    chinese_human_phrases = ["人工客服", "转人工", "人工服务"]

    def apply(self, state: CustomerServiceState) -> None:
        content = state.content.lower()

        if not state.final_reply or len(state.final_reply.strip()) < 10:
            self._mark_reply_handoff(state, "reply is empty or too short")

        if any(self._contains_term(content, term) for term in self.high_risk_terms):
            state.need_handoff_to_human = True
            state.handoff_type = "no_reply_handoff"
            state.handoff_reason = "high risk complaint or escalation"
        elif self._requests_human_support(content):
            self._mark_reply_handoff(state, "customer requested human support")

        if state.need_handoff_to_human and state.handoff_type == "no_handoff":
            self._mark_reply_handoff(state, "corrected inconsistent handoff state")

        if not state.need_handoff_to_human and state.handoff_type != "no_handoff":
            state.handoff_type = "no_handoff"
            state.handoff_reason = None

    @staticmethod
    def _contains_term(content: str, term: str) -> bool:
        if not term.isascii():
            return term in content
        return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", content) is not None

    def _requests_human_support(self, content: str) -> bool:
        return any(pattern.search(content) for pattern in self.human_patterns) or any(
            phrase in content for phrase in self.chinese_human_phrases
        )

    def _mark_reply_handoff(
        self,
        state: CustomerServiceState,
        reason: str,
    ) -> None:
        state.need_handoff_to_human = True
        if state.handoff_type == "no_handoff":
            state.handoff_type = "reply_handoff"
        state.handoff_reason = state.handoff_reason or reason
