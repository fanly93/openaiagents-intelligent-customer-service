from app.state.conversation_state import CustomerServiceState


class PromptAssembler:
    def build_instructions(
        self,
        state: CustomerServiceState,
        tool_guide: str = "",
        extra_instructions: str | None = None,
    ) -> str:
        slot_lines = [
            f"- {slot.name}: {slot.description} (required={slot.required})"
            for slot in state.slot_schema
        ]
        memory_lines = [
            f"- {memory.get('memory', memory)}" for memory in state.user_memories
        ]
        knowledge_lines = [
            f"- {item.get('text', item)}" for item in state.retrieved_knowledge
        ]

        parts = [
            "You are a professional customer service Agent for global ecommerce brands.",
            "Use the customer's language consistently across greeting, body, and signature.",
            "Extract configured slots before calling business tools when possible.",
            "Call handoff_to_human when the request needs human support.",
            "Never output THINK, ACTION, OBSERVE, or internal reasoning markers.",
            "## Slot Schema\n" + "\n".join(slot_lines) if slot_lines else "",
            "## User Memories\n" + "\n".join(memory_lines) if memory_lines else "",
            "## Retrieved Knowledge\n" + "\n".join(knowledge_lines)
            if knowledge_lines
            else "",
            "## Available Tools\n" + tool_guide if tool_guide else "",
            "## Tenant Instructions\n" + extra_instructions
            if extra_instructions
            else "",
        ]
        return "\n\n".join(part for part in parts if part)

    def build_user_message(self, state: CustomerServiceState) -> str:
        history = "\n".join(f"{turn.role}: {turn.content}" for turn in state.contexts)
        if history:
            return f"Current message:\n{state.content}\n\nConversation history:\n{history}"
        return state.content
