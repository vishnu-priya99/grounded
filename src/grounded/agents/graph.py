"""LangGraph wiring + the public ``answer()`` entry point.

The graph is deliberately linear — router -> retrieval -> table -> synthesis ->
verification. Each node no-ops when the router's decision makes it irrelevant.
Sequential execution keeps only one model in flight at a time, which matters on
a CPU-only machine, and makes the trace trivial to follow.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph

from grounded.agents.fallback import run_fallback
from grounded.agents.retrieval import run_retrieval
from grounded.agents.router import run_router
from grounded.agents.state import AgentState, Citation, SentenceCheck, TableAnswer
from grounded.agents.synthesis import run_synthesis
from grounded.agents.table import run_table
from grounded.agents.verification import _REFUSAL, run_verification
from grounded.config import settings


@dataclass
class AnswerResult:
    answer: str
    confidence: float
    intent: str = "doc_qa"
    refused: bool = False
    fallback_used: bool = False
    """True if this is an AI-computed guess (no answer key in the document),
    not a grounded, cited answer — render it visibly differently."""
    citations: list[Citation] = field(default_factory=list)
    tables: list[TableAnswer] = field(default_factory=list)
    checks: list[SentenceCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _traced(
    name: str, fn: Callable[[AgentState], AgentState]
) -> Callable[[AgentState], AgentState]:
    """Wrap an agent node so its progress prints live to the console as the
    graph runs. Each LLM call inside already prints its own line (see
    ``llm.py``); this brackets it with which agent is running and how long the
    whole step took, so a slow console never sits silent."""

    def wrapped(state: AgentState) -> AgentState:
        print(f"[grounded] -> {name}...", flush=True)
        t0 = time.perf_counter()
        out = fn(state)
        dt = time.perf_counter() - t0
        for note in out.get("notes", []):
            print(f"[grounded]      {note}", flush=True)
        print(f"[grounded] <- {name} done ({dt:.1f}s)", flush=True)
        return out

    return wrapped


def _after_verification(state: AgentState) -> str:
    """Route to the fallback agent only for a refused, item-targeted question
    — everything else (including a plain "not in your documents" on a
    non-numbered question) ends here, unchanged.

    Checking ``refused`` alone isn't enough: verification can also reach the
    exact same "I couldn't find this" answer via its low-confidence *hedge*
    path (partial citation coverage keeps ``refused`` False even though the
    text is a refusal) — so this checks the actual answer text, which is
    what synthesis/verification actually produce when nothing was grounded,
    regardless of which numeric branch got them there."""
    if not state["decision"].ordinals:
        return "end"
    if _REFUSAL in state.get("answer", ""):
        return "fallback"
    return "end"


def build_graph():
    g: StateGraph = StateGraph(AgentState)
    g.add_node("router", _traced("router", run_router))
    g.add_node("retrieval", _traced("retrieval", run_retrieval))
    g.add_node("table", _traced("table", run_table))
    g.add_node("synthesis", _traced("synthesis", run_synthesis))
    g.add_node("verification", _traced("verification", run_verification))
    g.add_node("fallback", _traced("fallback", run_fallback))

    g.add_edge(START, "router")
    g.add_edge("router", "retrieval")
    g.add_edge("retrieval", "table")
    g.add_edge("table", "synthesis")
    g.add_edge("synthesis", "verification")
    g.add_conditional_edges(
        "verification", _after_verification, {"fallback": "fallback", "end": END}
    )
    g.add_edge("fallback", END)
    return g.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def answer(
    question: str,
    *,
    history: str = "",
    source_files: list[str] | None = None,
) -> AnswerResult:
    t0 = time.perf_counter()
    print(
        f"\n[grounded] ==== {settings.llm_provider}/{settings.llm_model} "
        f"(router: {settings.router_model}) — {question!r} ====",
        flush=True,
    )
    state: AgentState = _graph().invoke(
        {
            "question": question,
            "history": history,
            "source_files": source_files,
            "notes": [],
        }
    )
    result = AnswerResult(
        answer=state.get("answer") or state.get("draft_answer") or "",
        confidence=state.get("confidence", 0.0),
        intent=state.get("intent", "doc_qa"),
        refused=state.get("refused", False),
        fallback_used=state.get("fallback_used", False),
        citations=state.get("citations", []),
        tables=state.get("tables", []),
        checks=state.get("checks", []),
        notes=state.get("notes", []),
    )
    status = (
        "FALLBACK (AI-computed, ungrounded)" if result.fallback_used
        else "REFUSED" if result.refused
        else f"confidence={result.confidence}"
    )
    print(
        f"[grounded] ==== done in {time.perf_counter() - t0:.1f}s — {status} ====\n",
        flush=True,
    )
    return result
