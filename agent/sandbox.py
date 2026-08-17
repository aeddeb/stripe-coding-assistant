"""Stripe test-mode sandbox execution: prove an answer by running it.

When an answer describes an executable payment flow, this module runs that
flow against Stripe's test mode and returns the trace — every API call with
the exact request payload sent and the full response returned, the status
transitions, and the events Stripe recorded (the same events a webhook
endpoint would receive, in order). Answer plus evidence.

Safety:
- Executes ONLY with a test-mode key (``sk_test_...``). Anything else is
  refused before any call is made.
- Flows are whitelisted: a question can only trigger a vetted, hardcoded
  sequence — never arbitrary API calls derived from model output.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import stripe


@dataclass
class TraceStep:
    """One API call in an executed flow."""

    call: str  # e.g. "PaymentIntent.create"
    request: dict  # the exact params sent to the API
    response: dict  # the full API response, null fields dropped
    object_id: str
    status: str
    note: str = ""


@dataclass
class SandboxTrace:
    flow: str
    title: str
    steps: list[TraceStep] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # chronological
    duration_ms: int = 0
    error: str | None = None


class SandboxKeyError(RuntimeError):
    """Raised when no usable test-mode key is configured."""


def _client() -> stripe.StripeClient:
    key = os.getenv("STRIPE_SECRET_KEY", "")
    if not key:
        raise SandboxKeyError("STRIPE_SECRET_KEY is not set.")
    if not key.startswith("sk_test_"):
        raise SandboxKeyError(
            "Refusing to run: STRIPE_SECRET_KEY is not a test-mode key "
            "(sk_test_...). The sandbox never executes against live mode."
        )
    return stripe.StripeClient(key)


def _response_dict(obj: stripe.StripeObject) -> dict:
    """The full API response as plain JSON-able data. Stripe pads responses
    with many null fields; drop them so the trace stays readable without
    hiding anything that carries a value."""
    return _strip_nulls(obj.to_dict(for_json=True))


def _strip_nulls(value):
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nulls(v) for v in value]
    return value


def _hold_and_capture_request(amount_cents: int) -> dict:
    """The exact PaymentIntent.create payload the flow sends."""
    return {
        "amount": amount_cents,
        "currency": "usd",
        "capture_method": "manual",  # hold now, capture later
        "confirm": True,
        "payment_method": "pm_card_visa",  # Stripe's test Visa
        "automatic_payment_methods": {
            "enabled": True,
            "allow_redirects": "never",
        },
    }


def preview_hold_and_capture(amount_cents: int = 5000) -> list[dict]:
    """The calls the flow will make, with their exact payloads — so the
    trace can be shown to the user before anything executes."""
    return [
        {
            "call": "PaymentIntent.create",
            "request": _hold_and_capture_request(amount_cents),
            "note": "places the hold on a test card",
        },
        {
            "call": "PaymentIntent.capture",
            "request": {"payment_intent": "<id returned by step 1>"},
            "note": "captures the held funds",
        },
    ]


def run_hold_and_capture(amount_cents: int = 5000) -> SandboxTrace:
    """Place a hold on a card, then capture it — the classic
    authorize-now-capture-on-shipment flow."""
    trace = SandboxTrace(
        flow="hold_and_capture",
        title=f"Hold ${amount_cents / 100:.2f} on a test card, then capture it",
    )
    started = time.perf_counter()
    window_start = int(time.time()) - 1
    try:
        client = _client()

        create_params = _hold_and_capture_request(amount_cents)
        intent = client.v1.payment_intents.create(params=create_params)
        trace.steps.append(
            TraceStep(
                call="PaymentIntent.create",
                request=create_params,
                response=_response_dict(intent),
                object_id=intent.id,
                status=intent.status,
                note="money is held on the card, not yet moved",
            )
        )

        captured = client.v1.payment_intents.capture(intent.id)
        trace.steps.append(
            TraceStep(
                call="PaymentIntent.capture",
                # Capture takes the id in the URL path; no body params sent.
                request={"payment_intent": intent.id},
                response=_response_dict(captured),
                object_id=captured.id,
                status=captured.status,
                note="the held funds are now captured",
            )
        )

        # The Events API is eventually consistent — give the capture
        # events a moment to land before reading the record.
        time.sleep(2)
        trace.events = _events_for(client, intent.id, window_start)
    except SandboxKeyError as error:
        trace.error = str(error)
    except stripe.StripeError as error:
        trace.error = f"Stripe API error: {getattr(error, 'user_message', None) or error}"
    trace.duration_ms = round((time.perf_counter() - started) * 1000)
    return trace


def _events_for(
    client: stripe.StripeClient, payment_intent_id: str, since_epoch: int
) -> list[dict]:
    """Events Stripe recorded for this flow, oldest first — what a webhook
    endpoint subscribed to these event types would have received."""
    events = client.v1.events.list(
        params={"created": {"gte": since_epoch}, "limit": 50}
    )
    related = []
    for event in events.data:
        obj = event.data.object
        if (
            getattr(obj, "id", None) == payment_intent_id
            or getattr(obj, "payment_intent", None) == payment_intent_id
        ):
            related.append({"type": event.type, "id": event.id, "created": event.created})
    # Same-second timestamps are common in a fast flow; break ties with
    # the canonical PaymentIntent lifecycle order.
    lifecycle = [
        "payment_intent.created",
        "payment_intent.amount_capturable_updated",
        "charge.succeeded",
        "payment_intent.succeeded",
        "charge.captured",
    ]
    def order(e: dict) -> tuple:
        pos = lifecycle.index(e["type"]) if e["type"] in lifecycle else len(lifecycle)
        return (e["created"], pos, e["id"])
    return sorted(related, key=order)


# --- Flow routing ----------------------------------------------------------

# Whitelisted flows: matcher -> runner. A question that matches nothing gets
# no sandbox offer — concept questions stay answer-only.
FLOWS = {
    "hold_and_capture": {
        "pattern": re.compile(
            r"hold|captur|authoriz|auth.{0,20}(then|later|ship)"
            r"|charge.{0,40}(later|when|after|ship)|uncaptured",
            re.IGNORECASE,
        ),
        "run": run_hold_and_capture,
        "preview": preview_hold_and_capture,
        "label": "place a hold, then capture it",
    },
}


def match_flow(question: str) -> str | None:
    """Return the flow key this question can prove, or None."""
    for key, flow in FLOWS.items():
        if flow["pattern"].search(question):
            return key
    return None


def preview_flow(key: str) -> list[dict]:
    """The flow's planned calls and payloads, without executing anything."""
    return FLOWS[key]["preview"]()


def run_flow(key: str) -> SandboxTrace:
    return FLOWS[key]["run"]()


if __name__ == "__main__":
    result = run_hold_and_capture()
    if result.error:
        raise SystemExit(result.error)
    for i, step in enumerate(result.steps, start=1):
        print(f"{i}. {step.call} -> {step.object_id} [{step.status}]  ({step.note})")
    print("events:", " -> ".join(e["type"] for e in result.events) or "(none yet)")
    print(f"({result.duration_ms} ms)")
