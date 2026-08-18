"""Streamlit chat UI for the Stripe Coding Assistant.

Run locally:  make app   (or: uv run --env-file .env streamlit run app/app.py)

Each question is answered independently — the visible chat history is for
reading back, not model memory. A question flows through
``services.answer.answer_routed`` (scope-gate/split → retrieve → grounded
prompt → generate) and renders with its citations. Every question, answer,
router verdict, model cost, and thumbs rating is logged to Postgres
(``app`` schema) for the monitoring dashboard.

Abuse limits for a public demo: question length is capped, each session has
a question budget, a database-backed daily ceiling covers all visitors, and
anything shaped like a Stripe API key is redacted before the question is
stored or sent anywhere.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

# Streamlit puts the script's own folder (app/) on sys.path, not the project
# root — make the project root importable so `services` resolves no matter
# how the app is launched (make app, plain streamlit run, or a cloud host).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from psycopg.types.json import Jsonb

from services import db
from agent.sandbox import FLOWS, SandboxTrace, match_flow, preview_flow, run_flow
from services.answer import (
    CODE_LANGUAGES,
    PROMPT_VARIANTS,
    Answer,
    answer_routed,
    language_instruction,
)

# Winner of the LLM evaluation (see README): best faithfulness (1.93/2)
# and relevance (2.00/2) across the ground-truth question set. The
# code-language rules are appended per question at serving time — the
# evaluation compared the bare variants.
SERVING_PROMPT = PROMPT_VARIANTS["v3-explained-summary"]
from services.retrieval import RetrievalConfig

LOGGER = logging.getLogger(__name__)

# --- Configuration ---------------------------------------------------------

# Retrieval settings served to users. Placeholder until the retrieval
# evaluation picks a winner; keyword/any is the implemented mode today and
# the forgiving match keeps wordy questions from returning nothing.
# Winner of the retrieval eval grid (see README "Retrieval evaluation"):
# best MRR overall (0.49) and tied-best hit rate at its k (0.73).
SERVING_CONFIG = RetrievalConfig(mode="vector", top_k=10)

MAX_QUESTION_CHARS = 1500
MAX_QUESTIONS_PER_SESSION = 25
# Hard, refresh-proof ceiling: counted from the database, shared by all
# visitors, resets at midnight. Protects the LLM budget on a public demo.
MAX_QUESTIONS_PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY", "200"))
# Burst brake, also database-counted: caps how fast the daily budget can
# drain when a script hammers the demo. Refreshing the page doesn't reset
# it, because the counter lives in Postgres, not the session.
MAX_QUESTIONS_PER_MINUTE = int(os.getenv("MAX_QUESTIONS_PER_MINUTE", "10"))
# Per-session pacing: minimum seconds between two questions from one tab.
QUESTION_COOLDOWN_SECONDS = int(os.getenv("QUESTION_COOLDOWN_SECONDS", "5"))

EXAMPLE_QUESTIONS = [
    "How do I charge a customer $50 but only capture the money when I ship?",
    "What's the difference between Checkout and Payment Links?",
    "How should I handle the checkout.session.completed webhook?",
]

# Stripe-style secrets (sk_test_..., pk_live_..., whsec_...). Users sometimes
# paste real keys into chat bots; strip them before the question is logged,
# displayed back, or sent to any model.
SECRET_PATTERN = re.compile(
    r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]+|\bwhsec_[A-Za-z0-9]+"
)


# --- Postgres helpers ------------------------------------------------------
# Short-lived connection per operation: Streamlit reruns this script in
# worker threads, and a shared cached connection is not thread-safe. The
# monitoring writes are tiny, so per-call connections cost little.


def _execute(sql: str, params: tuple = ()) -> None:
    with db.connect() as conn:  # commits on clean exit
        conn.execute(sql, params)


def _fetch_one(sql: str, params: tuple = ()):
    with db.connect() as conn:
        return conn.execute(sql, params).fetchone()


def _conversation_id() -> uuid.UUID:
    """Create the conversation row lazily, on the first question."""
    if "conversation_id" not in st.session_state:
        cid = uuid.uuid4()
        _execute("INSERT INTO app.conversations (id) VALUES (%s)", (cid,))
        st.session_state.conversation_id = cid
    return st.session_state.conversation_id


def _log_message(question: str, ans: Answer, error: str | None) -> int | None:
    """Record the exchange for monitoring. Never breaks the chat on failure."""
    # Which model answered and what it cost. Absent when no answer was
    # generated — a refused or failed question never reached a model.
    usage = ans.usage
    try:
        row = _fetch_one(
            """
            INSERT INTO app.messages
                (conversation_id, question, answer, route, retrieval_config,
                 retrieved_chunks, latency_ms, error, router,
                 provider, model, prompt_tokens, completion_tokens, cached)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                _conversation_id(),
                question,
                ans.text,
                ans.route,
                Jsonb(asdict(SERVING_CONFIG)),
                Jsonb(
                    [
                        {"chunk_id": h.chunk_id, "page_url": h.page_url, "score": h.score}
                        for h in ans.hits
                    ]
                ),
                ans.latency_ms,
                error,
                Jsonb(ans.router) if ans.router else None,
                usage.provider if usage else None,
                usage.model if usage else None,
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
                usage.cached if usage else None,
            ),
        )
        return row[0]
    except Exception:
        LOGGER.exception("failed to log message")
        return None


def _daily_capped() -> bool:
    """True when today's global question budget is spent. Fails open — a
    monitoring-DB hiccup should not lock users out."""
    try:
        row = _fetch_one(
            "SELECT count(*) FROM app.messages "
            "WHERE asked_at >= date_trunc('day', now())"
        )
        return row[0] >= MAX_QUESTIONS_PER_DAY
    except Exception:
        LOGGER.exception("daily-cap check failed — allowing question")
        return False


def _burst_capped() -> bool:
    """True when questions (from anyone) arrived faster than the per-minute
    budget — slows scripted hammering without touching real users."""
    try:
        row = _fetch_one(
            "SELECT count(*) FROM app.messages "
            "WHERE asked_at >= now() - interval '1 minute'"
        )
        return row[0] >= MAX_QUESTIONS_PER_MINUTE
    except Exception:
        LOGGER.exception("burst-cap check failed — allowing question")
        return False


def _cooldown_remaining() -> int:
    """Seconds this session must still wait before its next question."""
    last = st.session_state.get("last_question_at")
    if last is None:
        return 0
    return max(0, QUESTION_COOLDOWN_SECONDS - int(time.monotonic() - last))


def _save_feedback(message_id: int, widget_key: str) -> None:
    value = st.session_state.get(widget_key)  # thumbs: 0 = down, 1 = up
    if value is None:
        return
    try:
        _execute(
            """
            INSERT INTO app.feedback (message_id, rating)
            VALUES (%s, %s)
            ON CONFLICT (message_id)
            DO UPDATE SET rating = excluded.rating, given_at = now()
            """,
            (message_id, 1 if value == 1 else -1),
        )
    except Exception:
        LOGGER.exception("failed to save feedback")


# --- Answering -------------------------------------------------------------

# Display names for the providers in the fallback chain. Which one answers
# is decided at request time, so the answer names the model that actually
# served it rather than a configured default.
PROVIDER_LABELS = {"gemini": "Gemini", "groq": "Groq", "openai": "OpenAI"}


def _model_note(usage) -> str | None:
    """Attribution line for an answer, or None when no model was called
    (refusals and errors never reach one)."""
    if usage is None:
        return None
    provider = PROVIDER_LABELS.get(usage.provider, usage.provider.title())
    return f"Generated with {usage.model} ({provider})"


def _answer_and_log(question: str) -> dict:
    """Run the pipeline, log the exchange, return a renderable entry."""
    error = None
    prompt = SERVING_PROMPT + language_instruction(
        st.session_state.get("code_language", CODE_LANGUAGES[0])
    )
    try:
        with db.connect() as conn:
            ans = answer_routed(
                question, SERVING_CONFIG, conn, system_prompt=prompt
            )
    except Exception as exc:
        LOGGER.exception("answer pipeline failed")
        error = str(exc)
        ans = Answer(
            text="Something went wrong while answering. Please try again in a moment.",
            route="error",
            hits=[],
            latency_ms=0,
        )
    return {
        "role": "assistant",
        "text": ans.text,
        "route": ans.route,
        "citations": [
            {
                "page_title": h.page_title,
                "page_url": h.page_url,
                "heading_path": h.heading_path,
            }
            for h in ans.hits
        ],
        "message_id": _log_message(question, ans, error),
        "model_note": _model_note(ans.usage),
        # Executable flows get a sandbox offer; concept questions stay
        # answer-only. Matching is a whitelist, never model-driven.
        "sandbox_flow": match_flow(question) if ans.route == "docs" else None,
    }


# --- Rendering -------------------------------------------------------------


def _render_answer(entry: dict) -> None:
    st.markdown(entry["text"])
    citations = entry.get("citations") or []
    if citations:
        with st.expander(f"Sources ({len(citations)})"):
            for i, c in enumerate(citations, start=1):
                line = f"**[{i}]** [{c['page_title'] or c['page_url']}]({c['page_url']})"
                if c.get("heading_path"):
                    line += f" — {c['heading_path']}"
                st.markdown(line)
    flow_key = entry.get("sandbox_flow")
    if flow_key:
        entry_key = entry.get("message_id") or "draft"
        with st.expander("🧪 Stripe Sandbox — run the sample code in Stripe Test Mode"):
            st.caption(
                "This executes the recommended flow "
                f"(**{FLOWS[flow_key]['label']}**) against Stripe's test-mode "
                "sandbox — real API calls, no real money. The payloads below "
                "are exactly what will be sent."
            )
            for i, planned in enumerate(preview_flow(flow_key), start=1):
                st.markdown(f"**{i}. `{planned['call']}`** — {planned['note']}")
                st.json(planned["request"], expanded=True)
            trace_key = f"sandbox_trace_{entry_key}"
            if st.button(
                "▶ Run it", key=f"sandbox_btn_{entry_key}"
            ) and trace_key not in st.session_state:
                with st.spinner("Executing against Stripe test mode…"):
                    st.session_state[trace_key] = run_flow(flow_key)
            trace = st.session_state.get(trace_key)
            if trace:
                _render_trace(trace)
    if entry.get("model_note"):
        st.caption(entry["model_note"])
    if entry.get("message_id"):
        key = f"feedback_{entry['message_id']}"
        st.feedback(
            "thumbs", key=key, on_change=_save_feedback, args=(entry["message_id"], key)
        )


def _render_trace(trace: SandboxTrace) -> None:
    if trace.error:
        st.error(trace.error)
        return
    # The request payloads are already on screen (the pre-run preview), so
    # the trace shows what came back for each call.
    st.markdown(f"**{trace.title}** — executed:")
    for i, step in enumerate(trace.steps, start=1):
        st.markdown(
            f"**{i}. `{step.call}`** → `{step.object_id}` — "
            f"status **`{step.status}`**, {step.note}"
        )
        st.caption("Response returned (null fields omitted)")
        st.json(step.response, expanded=False)
    if trace.events:
        st.markdown("**Events Stripe recorded** (what your webhook would receive):")
        st.markdown(" → ".join(f"`{e['type']}`" for e in trace.events))
    st.caption(
        f"Executed in {trace.duration_ms} ms against Stripe test mode. "
        "No real money moved."
    )


# --- Page ------------------------------------------------------------------

st.set_page_config(page_title="Stripe Coding Assistant", page_icon="💳")
st.title("💳 Stripe Coding Assistant")
st.caption(
    "Ask how to build with Stripe. Answers come from the official docs, "
    "with citations you can check."
)

with st.sidebar:
    st.subheader("What can I ask?")
    st.markdown(
        "Anything about **integrating Stripe**: payments, Checkout, "
        "webhooks, subscriptions, refunds, disputes, testing."
    )
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- *{q}*")
    st.divider()
    st.selectbox(
        "Code samples in",
        CODE_LANGUAGES,
        key="code_language",
        help="Answers write their code samples in this language, "
        "whatever language the underlying docs page uses.",
    )
    st.divider()
    st.caption(
        "Answers are generated from retrieved documentation and can be "
        "imperfect — sources are linked so you can verify. "
        "Don't paste API keys or personal data."
    )

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        if entry["role"] == "user":
            st.markdown(entry["text"])
        else:
            _render_answer(entry)

question = st.chat_input(
    "Ask about integrating Stripe…", max_chars=MAX_QUESTION_CHARS
)

if question:
    # Cheap, local checks first — they cost nothing and can run before the
    # question is echoed.
    asked = sum(1 for e in st.session_state.history if e["role"] == "user")
    if asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning("This session has hit its question limit — refresh the page to start fresh.")
        st.stop()
    wait = _cooldown_remaining()
    if wait:
        st.info(f"One at a time — you can ask again in {wait}s.")
        st.stop()

    question = SECRET_PATTERN.sub("[redacted key]", question).strip()
    # Echo the question immediately. The rate-limit checks below query the
    # database — against a remote DB that round-trip is a visible pause, and
    # an unechoed question reads as "nothing happened".
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Stripe docs…"):
            if _daily_capped():
                st.warning(
                    "The demo has answered its daily budget of questions — "
                    "please come back tomorrow."
                )
                st.stop()
            if _burst_capped():
                st.warning("The demo is getting a burst of traffic — try again in a minute.")
                st.stop()
            st.session_state.last_question_at = time.monotonic()
            entry = _answer_and_log(question)
        _render_answer(entry)
    st.session_state.history.append({"role": "user", "text": question})
    st.session_state.history.append(entry)
