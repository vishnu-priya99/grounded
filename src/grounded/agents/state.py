"""Typed state passed between agents in the LangGraph runtime."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

from grounded.index.store import Retrieved

Intent = Literal["doc_qa", "table_qa", "mixed", "code_qa", "chitchat"]


class RouterDecision(BaseModel):
    """Structured output of the Router/Planner agent."""

    intent: Intent
    rewritten_question: str = Field(
        description="The user's question rewritten to stand alone, with pronouns and "
        "references from the conversation resolved."
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="1-3 focused search queries. For a simple question, just one.",
    )
    needs_tables: bool = Field(
        description="True if answering requires computing over spreadsheet data."
    )
    ordinals: list[int] = Field(
        default_factory=list,
        description="Every enumerated item targeted by position — question "
        "numbers, 'Nth', an 'and'/comma-separated list ('question 2 and 3' -> "
        "[2, 3], '10th and 11th' -> [10, 11], 'questions 5, 7, and 9' -> "
        "[5, 7, 9]). Use -1 for 'the last one'. Empty list if the question "
        "isn't positional at all.",
    )
    broad: bool = Field(
        default=False,
        description="True if answering needs to survey the WHOLE document rather "
        "than a few passages — 'what topics does this cover', 'list every X', "
        "'summarize the document', 'how many questions are about Y'. False for a "
        "question answerable from one or two specific passages.",
    )
    reasoning: str = ""

    @field_validator("ordinals", mode="before")
    @classmethod
    def _blank_ordinals_is_empty(cls, v: object) -> object:
        """Small models occasionally emit null/"" instead of [] for "none"."""
        if v is None or (isinstance(v, str) and v.strip().lower() in {"", "null", "none"}):
            return []
        return v


class SynthesisResult(BaseModel):
    """Structured output of the Synthesis agent — replaces the older approach
    of asking the model to reply with one exact magic string ("I couldn't
    find this in your documents.") to signal a refusal. That was fragile: any
    rephrasing of the refusal (a hedge, a different wording) went undetected
    by an exact-string check downstream. A structured boolean can't be
    rephrased out from under the check."""

    found: bool = Field(
        description="True if the context actually contains the answer. False "
        "if it doesn't — including when the context shows a multiple-choice "
        "question's options but never marks one as correct."
    )
    answer: str = Field(
        default="",
        description="The grounded answer, citing sources with [n] markers, "
        "using ONLY the context. Empty when found is false.",
    )


class Citation(BaseModel):
    n: int
    source_file: str
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    snippet: str = ""
    source_text: str = Field(default="", exclude=True)
    """Full context the synthesiser saw for this marker; used by verification."""
    support: float | None = None
    """Entailment score from the verification agent, 0-1."""

    @property
    def label(self) -> str:
        loc = f"p.{self.page}" if self.page is not None else (
            " > ".join(self.section_path) or "start"
        )
        return f"{self.source_file} ({loc})"


class TableAnswer(BaseModel):
    question: str
    sql: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[list] = Field(default_factory=list)
    error: str = ""
    attempts: int = 0

    def as_markdown(self, limit: int = 20) -> str:
        if self.error:
            return f"_SQL error: {self.error}_"
        if not self.columns:
            return "_no result_"
        head = "| " + " | ".join(map(str, self.columns)) + " |"
        sep = "| " + " | ".join(["---"] * len(self.columns)) + " |"
        body = [
            "| " + " | ".join("" if c is None else str(c) for c in row) + " |"
            for row in self.rows[:limit]
        ]
        more = f"\n_({len(self.rows) - limit} more rows)_" if len(self.rows) > limit else ""
        return "\n".join([head, sep, *body]) + more


class SentenceCheck(BaseModel):
    sentence: str
    supported: bool
    cited: list[int] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    history: str
    source_files: list[str] | None

    # router
    decision: RouterDecision
    intent: Intent

    # gather
    passages: list[Retrieved]
    tables: list[TableAnswer]

    # compose / check
    draft_answer: str
    answer: str
    citations: list[Citation]
    checks: list[SentenceCheck]
    confidence: float
    refused: bool
    fallback_used: bool
    """True when the grounded pipeline refused and the fallback agent stepped
    in with an AI-computed (not document-sourced) answer instead."""
    notes: Annotated[list[str], operator.add]
