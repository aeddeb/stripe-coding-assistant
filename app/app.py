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
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

# Streamlit puts the script's own folder (app/) on sys.path, not the project
# root — make the project root importable so `services` resolves no matter
# how the app is launched (make app, plain streamlit run, or a cloud host).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from psycopg.types.json import Jsonb

from services import db
from agent.sandbox import (
    FLOWS,
    SandboxTrace,
    coerce_params,
    flow_catalog,
    match_flow,
    preview_flow,
    run_flow,
)
from services.answer import (
    CODE_LANGUAGES,
    FORMATTING_INSTRUCTION,
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

# Chosen to show what this assistant does that a generic chatbot does not:
# the first offers a runnable sandbox demonstration, the second is split by
# the router into two retrievals, the third lands on the asynchronous part
# of an integration where copy-pasted answers usually fail.
EXAMPLE_QUESTIONS = [
    "How do I place a hold on a customer's card and only capture it when I ship?",
    "What's the difference between Checkout and Payment Links, and which "
    "one should I use for a one-off invoice?",
    "How do I make sure I never fulfill an order twice when handling "
    "checkout.session.completed?",
]

# What the progress line says while each pipeline stage runs. The pipeline
# reports stage keys and the wording lives here, in the UI.
STAGE_LABELS = {
    "routing": "Understanding the question…",
    "retrieval": "Searching the Stripe docs…",
    "generation": "Writing a grounded answer…",
}

# What "this answer contains code" looks like: a fenced block. Inline
# backticks are not enough — a Dashboard walkthrough is full of them, and
# naming `capture_method` mid-sentence is not something you can execute.
# The leading spaces matter: answers put code inside numbered steps, so
# almost every real fence is indented and an anchored ^``` matches none.
CODE_FENCE = re.compile(r"^[ \t]*```", re.MULTILINE)

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


def _log_message(
    question: str, ans: Answer, error: str | None, sandbox_flow: str | None
) -> int | None:
    """Record the exchange for monitoring. Never breaks the chat on failure."""
    try:
        # Which model answered and what it cost. Absent when no answer was
        # generated — a refused or failed question never reached a model.
        usage = ans.usage
        row = _fetch_one(
            """
            INSERT INTO app.messages
                (conversation_id, question, answer, route, retrieval_config,
                 retrieved_chunks, latency_ms, error, router,
                 provider, model, prompt_tokens, completion_tokens, cached,
                 sandbox_flow, code_language)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s)
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
                sandbox_flow,
                # Which language the answer's code samples were asked for,
                # so the dashboard can see whether the selector is used.
                st.session_state.get("code_language"),
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


def _answer_and_log(
    question: str, on_stage: Callable[[str], None] | None = None
) -> dict:
    """Run the pipeline, log the exchange, return a renderable entry.

    ``on_stage`` is passed through to the pipeline, which calls it as each
    stage starts so the progress line can name the stage running.
    """
    error = None
    prompt = (
        SERVING_PROMPT
        + language_instruction(
            st.session_state.get("code_language", CODE_LANGUAGES[0])
        )
        + FORMATTING_INSTRUCTION
    )
    try:
        with db.connect() as conn:
            ans = answer_routed(
                question, SERVING_CONFIG, conn, system_prompt=prompt,
                sandbox_flows=flow_catalog(), on_stage=on_stage,
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
    # The frozen v3 prompt names the second section "Code sample" and
    # instructs that the structure be followed exactly, so an answer built
    # from Dashboard pages still files its click-steps under a heading that
    # promises code. No addendum wording reliably talks it out of that.
    # Whether code is present is mechanically decidable, so it is decided
    # here instead of asked twice — and from the same signal that gates the
    # sandbox, so the heading and the sandbox can never disagree.
    if not _has_code(ans.text):
        ans.text = ans.text.replace("**Code sample**", "**Steps**")
    # Decided once: the same value is logged for monitoring and rendered
    # in the UI, so the dashboard cannot disagree with what users saw.
    sandbox_flow = _sandbox_choice(question, ans)
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
        "message_id": _log_message(question, ans, error, sandbox_flow),
        "model_note": _model_note(ans.usage),
        # Which runnable demonstration to offer. The router chooses from a
        # fixed whitelist; the model can pick a flow but never write one.
        "sandbox_flow": sandbox_flow,
        "sandbox_params": ans.sandbox_params,
        # A stable key for this answer's widgets. Streamlit keys must be
        # unique per widget, and message_id is None whenever the monitoring
        # write failed — two such answers would collide on a shared key.
        "widget_key": uuid.uuid4().hex[:8],
    }


def _has_code(text: str) -> bool:
    """Whether an answer actually shows code the sandbox could stand in for."""
    return bool(CODE_FENCE.search(text or ""))


def _sandbox_choice(question: str, ans: Answer) -> str | None:
    """The flow to offer for this answer, or None.

    Normally the router picks it. If the router call itself failed, the
    question still got answered (the pipeline fails open) — fall back to
    keyword matching so an outage does not silently remove the sandbox.

    The answer gets the last word. The router chooses from the question
    alone, before any answer exists, so it cannot know the docs would
    answer with a Dashboard walkthrough instead of code. Offering to
    execute API calls under an answer that contains none demonstrates
    something the user was never told to do.
    """
    if ans.route != "docs":
        return None
    if not _has_code(ans.text):
        return None
    if ans.sandbox_flow:
        return ans.sandbox_flow
    if (ans.router or {}).get("error"):
        return match_flow(question)
    return None


# --- Rendering -------------------------------------------------------------


# Progressive rendering. Nothing streams out of the LLM — the answer is
# complete before it reaches here — so this only paces text that already
# exists, so a long answer fills in instead of landing as one wall. Quick
# on purpose: roughly a second for a full-length answer.
STREAM_WORDS_PER_CHUNK = 6
STREAM_CHUNK_SECONDS = 0.012


def _word_stream(text: str):
    """Yield a finished answer in small bursts, for ``st.write_stream``."""
    words = text.split(" ")
    for i in range(0, len(words), STREAM_WORDS_PER_CHUNK):
        yield " ".join(words[i : i + STREAM_WORDS_PER_CHUNK]) + " "
        time.sleep(STREAM_CHUNK_SECONDS)


def _group_citations(citations: list[dict]) -> list[dict]:
    """One entry per source page, keeping every [n] that pointed at it.

    Retrieval returns chunks, and a long documentation page usually supplies
    several of them, so the raw list repeats the same page under different
    numbers. The numbers are never reassigned: they are the markers the
    answer text cites, and they have to keep pointing at the same excerpts.
    """
    grouped: dict[str, dict] = {}
    for i, c in enumerate(citations, start=1):
        page = grouped.setdefault(
            c["page_url"],
            {
                "page_url": c["page_url"],
                "page_title": c.get("page_title"),
                "indices": [],
                "headings": [],
            },
        )
        page["indices"].append(i)
        page["headings"].append(c.get("heading_path") or "")
    return list(grouped.values())


def _render_answer(entry: dict, stream: bool = False) -> None:
    # Streamed only when the answer has just been generated; replaying the
    # history renders instantly.
    if stream:
        st.write_stream(_word_stream(entry["text"]))
    else:
        st.markdown(entry["text"])
    citations = entry.get("citations") or []
    if citations:
        pages = _group_citations(citations)
        with st.expander(f"Sources ({len(pages)})"):
            for page in pages:
                markers = "".join(f"[{i}]" for i in page["indices"])
                title = page["page_title"] or page["page_url"]
                line = f"**{markers}** [{title}]({page['page_url']})"
                # Named only when every excerpt from this page came from the
                # same section — otherwise one heading would stand in for
                # several and describe the others wrongly.
                if len(set(page["headings"])) == 1 and page["headings"][0]:
                    line += f" — {page['headings'][0]}"
                st.markdown(line)
    _render_sandbox(entry)
    if entry.get("model_note"):
        st.caption(entry["model_note"])
    if entry.get("message_id"):
        key = f"feedback_{entry['message_id']}"
        st.caption("Rate this answer — was it accurate and useful?")
        st.feedback(
            "thumbs", key=key, on_change=_save_feedback, args=(entry["message_id"], key)
        )


def _render_sandbox(entry: dict) -> None:
    """The sandbox offer for one answer, or a line saying why there isn't one.

    Saying nothing was the old behaviour, and it read as a bug: a question
    the sandbox cannot demonstrate looked identical to one it had failed on.
    """
    if entry.get("route") != "docs":
        return
    flow_key = entry.get("sandbox_flow")
    if not flow_key:
        st.caption(
            "This answer doesn't include code to run, so there's nothing to "
            "execute against the sandbox."
            if not _has_code(entry.get("text", ""))
            else "No runnable sandbox demo covers this topic yet."
        )
        return

    flow = FLOWS[flow_key]
    params = coerce_params(flow_key, entry.get("sandbox_params"))
    widget_key = entry.get("widget_key") or entry.get("message_id") or "draft"
    trace_key = f"sandbox_trace_{widget_key}"
    run_key = f"sandbox_btn_{widget_key}"
    # Streamlit builds an expander closed unless told otherwise, and every
    # click reruns the whole script — so the panel used to shut itself on
    # exactly the rerun that produced the trace, hiding the result the user
    # had just asked for. Both conditions are needed: on the run itself the
    # trace does not exist yet, and the button's click is readable here
    # because the rerun it triggers carries its value in session state.
    just_ran = bool(st.session_state.get(run_key))

    with st.expander(
        "🧪 Stripe Sandbox — run this against Stripe Test Mode",
        expanded=just_ran or trace_key in st.session_state,
    ):
        st.caption(
            f"This executes **{flow.label}** against Stripe's test-mode "
            "sandbox — real API calls, no real money. The payloads below are "
            "exactly what will be sent."
        )
        if params:
            st.caption(
                "Running with "
                + ", ".join(
                    f"{flow.params[name].label.lower()} **{value}**"
                    for name, value in params.items()
                )
            )
        # Written after the run button so a finished run can collapse the
        # plan it has replaced; the container holds the slot up here.
        plan_box = st.container()
        if st.button("▶ Run it", key=run_key):
            with st.spinner("Executing against Stripe test mode…"):
                st.session_state[trace_key] = run_flow(flow_key, params)
        trace = st.session_state.get(trace_key)
        with plan_box:
            for i, planned in enumerate(preview_flow(flow_key, params), start=1):
                if trace:
                    # The trace below shows each request beside what it
                    # returned, so the payloads here would only repeat it.
                    st.markdown(f"{i}. `{planned['call']}` — {planned['note']}")
                else:
                    st.markdown(f"**{i}. `{planned['call']}`** — {planned['note']}")
                    st.json(planned["request"], expanded=False)
        if trace:
            _render_trace(trace)


def _render_trace(trace: SandboxTrace) -> None:
    if trace.error:
        st.error(trace.error)
        return
    # Each step carries its own request, so the call is shown next to what
    # it returned. The pre-run plan above collapses to one line per step
    # once this exists, rather than repeating the payloads.
    st.markdown(f"**{trace.title}** — executed:")
    for i, step in enumerate(trace.steps, start=1):
        # A decline is the point of its flow, not a malfunction — mark it as
        # a result, with the error Stripe returned as the response body.
        arrow = "✕" if step.failed else "→"
        target = f" `{step.object_id}`" if step.object_id else ""
        st.markdown(
            f"**{i}. `{step.call}`** {arrow}{target} — "
            f"status **`{step.status}`**, {step.note}"
        )
        if step.link:
            st.markdown(f"[Open the Checkout page ↗]({step.link})")
        st.caption("Sent")
        st.json(step.request, expanded=False)
        st.caption(
            "Returned — the error Stripe replied with" if step.failed
            else "Returned (null fields omitted)"
        )
        st.json(step.response, expanded=False)
    if trace.events:
        st.markdown("**Events Stripe recorded** (what your webhook would receive):")
        st.markdown(" → ".join(f"`{e['type']}`" for e in trace.events))
    st.caption(
        f"Executed in {trace.duration_ms} ms against Stripe test mode. "
        "No real money moved."
    )


# --- Page ------------------------------------------------------------------

st.set_page_config(page_title="Stripe Integration Assistant", page_icon="💳")
st.title("💳 Stripe Integration Assistant")
st.caption(
    "Ask how to build with Stripe. Answers come from the official docs, "
    "with citations you can check."
)
# Kept in the main column on purpose: Streamlit collapses the sidebar on
# narrow screens, so a disclaimer down there is invisible to most visitors
# arriving from a phone.
st.caption(
    "Answers are generated from Stripe's public documentation and can be "
    "wrong or out of date — check the linked sources before relying on "
    "them. Not affiliated with or endorsed by Stripe."
)

# Directly under the title, where a visitor arriving from a link looks
# first — and where it explains the name above it. Collapsed, so it costs
# one line above the conversation.
with st.expander("About this project"):
    st.markdown(
        "Stripe's documentation is excellent but enormous, and it is "
        "organized by product — so one integration question is usually "
        "answered across several pages. Ask a general-purpose AI instead "
        "and it fills the gaps with parameter names that look right but do "
        "not exist.\n\n"
        "This assistant answers only from the official Stripe docs, and "
        "links the pages behind every claim so you can check it. When the "
        "answer is a payment flow you can run, it proves the answer by "
        "running that flow in Stripe's test sandbox and showing the real "
        "API calls, responses, and webhook events."
    )

with st.sidebar:
    st.subheader("What can I ask?")
    st.markdown(
        "Anything about **integrating Stripe**: payments, Checkout, "
        "webhooks, subscriptions, refunds, disputes, testing."
    )
    st.caption("Try one of these:")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        # A chat_input cannot be prefilled, so an example is submitted
        # directly — picked up below, where a typed question is read.
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_question = q
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
        "Each question is answered on its own — answers don't carry "
        "context from earlier questions."
    )
    st.caption(
        "Answers are generated from retrieved documentation and can be "
        "imperfect — sources are linked so you can verify. "
        "Don't paste API keys or personal data."
    )
    st.divider()
    st.caption(
        "Built by [Ali Eddeb](https://www.linkedin.com/in/ali-eddeb/) · "
        "[Source on GitHub](https://github.com/aeddeb/stripe-coding-assistant)"
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
# An example button in the sidebar submits its question through here, so a
# clicked example and a typed one follow exactly the same path.
question = question or st.session_state.pop("pending_question", None)

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
        # The rate-limit checks are database round-trips, but they either
        # stop the question or cost nothing to report — the progress line
        # opens once the question is actually going to be answered.
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
        status = st.status(STAGE_LABELS["routing"], expanded=False)
        entry = _answer_and_log(
            question,
            on_stage=lambda stage: status.update(
                label=STAGE_LABELS.get(stage, stage)
            ),
        )
        failed = entry["route"] == "error"
        status.update(
            label="Something went wrong" if failed else "Answered",
            state="error" if failed else "complete",
        )
        _render_answer(entry, stream=True)
    st.session_state.history.append({"role": "user", "text": question})
    st.session_state.history.append(entry)
