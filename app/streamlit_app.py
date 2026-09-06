"""Grounded — Streamlit chat UI.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded import llm  # noqa: E402
from grounded.agents.graph import AnswerResult, answer  # noqa: E402
from grounded.config import settings  # noqa: E402
from grounded.ingest.pipeline import delete_source_file, get_stores, ingest_paths  # noqa: E402
from grounded.memory import SessionMemory  # noqa: E402

st.set_page_config(page_title="Grounded", page_icon="📑", layout="centered")

UPLOADS = settings.data_dir / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()
if "chat" not in st.session_state:
    st.session_state.chat = []  # list[tuple[str, AnswerResult | str]]


@st.dialog("Remove this document?")
def _confirm_remove(name: str) -> None:
    st.write(
        f"**{name}** will be removed from the index (vector store and any SQL "
        "tables), and its file deleted if it was an upload. This can't be undone."
    )
    left, right = st.columns(2)
    if left.button("Remove", type="primary", use_container_width=True):
        n = delete_source_file(name, stores=get_stores())
        st.session_state["removed_msg"] = f"Removed {name} ({n} chunk(s))"
        st.rerun()
    if right.button("Cancel", use_container_width=True):
        st.rerun()


def _confidence_badge(result: AnswerResult) -> str:
    if result.fallback_used:
        return "🧮 AI-computed — not from your documents, no citations"
    if result.refused:
        return "🚫 declined — not found in your documents"
    c = result.confidence
    dot = "🟢" if c >= 0.75 else "🟡" if c >= settings.confidence_threshold else "🟠"
    return f"{dot} confidence {c:.0%}"


def _render_answer(result: AnswerResult) -> None:
    st.markdown(result.answer)
    st.caption(_confidence_badge(result))

    if result.citations:
        with st.expander(f"Sources ({len(result.citations)})"):
            for c in result.citations:
                sup = "" if c.support is None else f" · match {c.support:.0%}"
                st.markdown(f"**[{c.n}] {c.label}**{sup}")
                st.caption(c.snippet)

    with st.expander("How this was answered"):
        if result.tables:
            for t in result.tables:
                if t.sql:
                    st.code(t.sql, language="sql")
                if t.rows:
                    st.dataframe(
                        {col: [r[i] for r in t.rows] for i, col in enumerate(t.columns)},
                        use_container_width=True, hide_index=True,
                    )
                if t.error:
                    st.warning(f"SQL note: {t.error}")
        if result.checks:
            st.markdown("**Sentence support** — is each sentence backed by its cited source?")
            for chk in result.checks:
                mark = "✅ supported" if chk.supported else "⚠️ weak"
                st.caption(f"{mark} — {chk.sentence[:200]}")
        st.markdown("**Agent trace**")
        for note in result.notes:
            st.caption(f"• {note}")


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📑 Grounded")
    st.caption("Local-first document chat. Cites sources, checks its own work.")

    # Bumping this key after an ingest gives the uploader a fresh identity, so
    # Streamlit clears the just-processed files instead of leaving them (and the
    # Ingest button) sitting there as if nothing happened.
    upload_key = st.session_state.get("upload_key", 0)
    uploaded = st.file_uploader(
        "Add files",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "html", "htm", "txt", "md", "csv", "xlsx",
              "png", "jpg", "jpeg", "py", "js", "ts", "java", "go", "sql"],
        key=f"uploader_{upload_key}",
    )
    if uploaded and st.button("Ingest", type="primary", use_container_width=True):
        paths = []
        for uf in uploaded:
            dest = UPLOADS / uf.name
            dest.write_bytes(uf.getbuffer())
            paths.append(dest)
        lines: list[str] = []
        with st.status("Ingesting…", expanded=True) as status:
            for r in ingest_paths(paths):
                if r.ok:
                    extra = f", {r.n_tables} tables" if r.n_tables else ""
                    lines.append(
                        f"✅ {r.source_file} — {r.n_chunks} chunks{extra} ({r.seconds:.1f}s)"
                    )
                else:
                    lines.append(f"❌ {r.source_file} — {r.error}")
                status.write(lines[-1])
            status.update(label="Ingestion complete", state="complete")
        st.session_state["last_ingest"] = lines
        st.session_state["upload_key"] = upload_key + 1
        st.rerun()

    if st.session_state.get("last_ingest"):
        with st.expander("Last ingest", expanded=False):
            for line in st.session_state["last_ingest"]:
                st.write(line)

    stores = get_stores()
    docs = stores.vector.documents()
    if msg := st.session_state.pop("removed_msg", None):
        st.toast(msg)
    st.divider()
    st.subheader(f"Indexed ({len(docs)})")
    scope = st.multiselect("Restrict answers to", docs, default=[])
    for d in docs:
        c1, c2 = st.columns([6, 1])
        c1.caption(f"• {d}")
        if c2.button("✕", key=f"del_{d}", help=f"Remove {d} from the index"):
            _confirm_remove(d)

    st.divider()
    settings.fast_mode = st.toggle("Fast mode (skip rerank)", value=settings.fast_mode)
    settings.confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, settings.confidence_threshold, 0.05
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat = []
        st.session_state.memory.clear()
        st.rerun()

    if settings.llm_provider != "ollama":
        usage = llm.usage_summary()
        if usage["calls"]:
            st.caption(
                f"💰 Session spend: ${usage['cost_usd']:.4f} "
                f"({usage['calls']} calls, {usage['input_tokens']}+"
                f"{usage['output_tokens']} tok) — since this app started"
            )


# ---------------------------------------------------------------------------
# main chat
# ---------------------------------------------------------------------------
st.title("Chat")
if not get_stores().vector.documents():
    st.info("Upload some files in the sidebar to get started.")

for question, result in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if isinstance(result, str):
            st.markdown(result)
        else:
            _render_answer(result)

if prompt := st.chat_input("Ask across your documents…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Routing → retrieving → synthesising → verifying…"):
            result = answer(
                prompt,
                history=st.session_state.memory.context(),
                source_files=scope or None,
            )
        _render_answer(result)
    st.session_state.memory.add(prompt, result.answer)
    st.session_state.chat.append((prompt, result))
