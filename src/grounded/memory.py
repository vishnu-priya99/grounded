"""Per-session conversation memory.

A rolling buffer of recent turns plus an LLM-maintained running summary of older
turns. The router consumes :meth:`context` to rewrite follow-up questions into
standalone ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from grounded.llm import chat

_KEEP_VERBATIM = 4  # most recent turns kept word-for-word


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class SessionMemory:
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""

    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question, answer))
        if len(self.turns) > _KEEP_VERBATIM:
            self._fold_oldest()

    def _fold_oldest(self) -> None:
        old = self.turns.pop(0)
        prompt = [
            {
                "role": "system",
                "content": "You maintain a terse running summary of a conversation "
                "between a user and a document-chat assistant. Keep it under 120 words. "
                "Preserve entities, filenames, and facts the user may refer back to.",
            },
            {
                "role": "user",
                "content": f"Current summary:\n{self.summary or '(none)'}\n\n"
                f"New exchange to fold in:\nUser: {old.question}\n"
                f"Assistant: {old.answer[:500]}\n\nUpdated summary:",
            },
        ]
        try:
            self.summary = chat(prompt, temperature=0.0).strip()
        except Exception:  # pragma: no cover - memory is best-effort
            self.summary = (self.summary + f" | Q: {old.question}").strip(" |")

    def context(self) -> str:
        parts: list[str] = []
        if self.summary:
            parts.append(f"Conversation so far: {self.summary}")
        for t in self.turns:
            parts.append(f"User: {t.question}\nAssistant: {t.answer[:400]}")
        return "\n\n".join(parts)

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
