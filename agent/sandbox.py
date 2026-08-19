"""Stripe test-mode sandbox execution: prove an answer by running it.

When an answer describes an executable payment flow, this module runs that
flow against Stripe's test mode and returns the trace — every API call with
the exact request payload sent and the full response returned, the status
transitions, and the events Stripe recorded (the same events a webhook
endpoint would receive, in order). Answer plus evidence.

Safety is structural, not behavioural:

- Executes ONLY with a test-mode key (``sk_test_...``). Anything else is
  refused before any call is made.
- Flows are whitelisted. A question can only trigger a vetted, hardcoded
  call sequence — the model never constructs an API call.
- The model may fill a flow's parameters, but those are numbers and fixed
  choices only, each clamped to a safe range here. No text the model wrote
  is ever sent to the Stripe API.

Every flow declares its planned calls (``plan``) separately from executing
them (``run``), and both build their payloads from the same helper — so
what the user is shown before clicking is what actually gets sent.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import stripe

# Fixed for every flow. Currency interacts with account settings and price
# creation in ways that can fail at demo time; the flows exist to teach the
# call sequence, not currency handling.
CURRENCY = "usd"
TEST_CARD = "pm_card_visa"
# Stripe's test card that succeeds, then immediately gets disputed as fraud.
DISPUTE_CARD = "pm_card_createDispute"


@dataclass
class TraceStep:
    """One API call in an executed flow."""

    call: str  # e.g. "PaymentIntent.create"
    request: dict  # the exact params sent to the API
    response: dict  # the full API response, null fields dropped
    object_id: str
    status: str
    note: str = ""
    link: str | None = None  # a URL from the response worth clicking
    failed: bool = False  # an expected failure (a decline) is still a result


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


def _response_dict(obj) -> dict:
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


def _error_dict(error: stripe.StripeError) -> dict:
    """A declined card is a result, not a crash — record what Stripe said."""
    err = getattr(error, "error", None)
    fields = {
        "type": getattr(err, "type", None),
        "code": getattr(err, "code", None),
        "decline_code": getattr(err, "decline_code", None),
        "message": getattr(err, "message", None) or str(error),
        "param": getattr(err, "param", None),
    }
    return {k: v for k, v in fields.items() if v is not None}


# --- Parameters ------------------------------------------------------------
# A flow's parameters are the only part of a sandbox run the language model
# influences. Keeping them to numbers and fixed choices — never free text —
# is what makes model involvement safe: the worst a hostile instruction
# hidden in a retrieved document can achieve is a different vetted flow
# running at a different dollar amount.


@dataclass(frozen=True)
class ParamSpec:
    label: str
    default: Any
    kind: str = "int"  # "int" | "choice"
    minimum: int = 50  # 50 cents — Stripe's minimum charge
    maximum: int = 1_000_000  # $10,000
    choices: tuple[str, ...] = ()

    def coerce(self, value: Any) -> Any:
        """Return a safe value for ``value``, falling back to the default.
        Never raises: bad model output degrades to the default, it does not
        break the run."""
        if self.kind == "choice":
            return value if value in self.choices else self.default
        try:
            number = int(value)
        except (TypeError, ValueError):
            return self.default
        return max(self.minimum, min(self.maximum, number))


def coerce_params(flow_key: str, raw: dict | None) -> dict:
    """Validate model-supplied parameters against a flow's spec. Unknown
    keys are dropped; missing and invalid ones fall back to defaults."""
    specs = FLOWS[flow_key].params
    raw = raw or {}
    return {name: spec.coerce(raw.get(name, spec.default)) for name, spec in specs.items()}


def _money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


# --- Execution helper ------------------------------------------------------


class _Run:
    """Accumulates a trace while a flow executes, and remembers the object
    ids the flow touched so the event lookup afterwards knows what to match."""

    def __init__(self, flow: str, title: str):
        self.trace = SandboxTrace(flow=flow, title=title)
        self.client = _client()
        self.ids: set[str] = set()
        self.window_start = int(time.time()) - 1

    def record(
        self,
        call: str,
        request: dict,
        response,
        note: str = "",
        link: str | None = None,
    ):
        obj_id = getattr(response, "id", "") or ""
        if obj_id:
            self.ids.add(obj_id)
        self.trace.steps.append(
            TraceStep(
                call=call,
                request=request,
                response=_response_dict(response),
                object_id=obj_id,
                status=str(getattr(response, "status", "") or "ok"),
                note=note,
                link=link,
            )
        )
        return response

    def record_failure(self, call: str, request: dict, error: stripe.StripeError, note: str = ""):
        """Record an API error as a step. Used by flows whose whole point is
        to show a failure — a declined card is the answer, not an outage."""
        payload = _error_dict(error)
        self.trace.steps.append(
            TraceStep(
                call=call,
                request=request,
                response=payload,
                object_id=payload.get("code", "") or "",
                status=payload.get("decline_code") or payload.get("code") or "error",
                note=note,
                failed=True,
            )
        )

    def collect_events(self, settle_seconds: float = 2.0):
        """The Events API is eventually consistent — give the flow's events a
        moment to land before reading the record."""
        if not self.ids:
            return
        time.sleep(settle_seconds)
        self.trace.events = _events_for(self.client, self.ids, self.window_start)


def _events_for(
    client: stripe.StripeClient, object_ids: set[str], since_epoch: int
) -> list[dict]:
    """Events Stripe recorded for this flow, oldest first — what a webhook
    endpoint subscribed to these event types would have received."""
    events = client.v1.events.list(params={"created": {"gte": since_epoch}, "limit": 100})
    related = []
    for event in events.data:
        obj = event.data.object
        # An event belongs to the flow when its object is one the flow
        # created, or points at one (a charge names its payment_intent, an
        # invoice names its subscription and customer).
        candidates = {
            getattr(obj, "id", None),
            getattr(obj, "payment_intent", None),
            getattr(obj, "subscription", None),
            getattr(obj, "customer", None),
            getattr(obj, "setup_intent", None),
        }
        if candidates & object_ids:
            related.append({"type": event.type, "id": event.id, "created": event.created})
    # Same-second timestamps are common in a fast flow; break ties with the
    # canonical lifecycle order so the sequence reads correctly.
    lifecycle = [
        "customer.created",
        "payment_method.attached",
        "customer.updated",
        "setup_intent.created",
        "setup_intent.succeeded",
        "payment_intent.created",
        "payment_intent.amount_capturable_updated",
        "charge.succeeded",
        "charge.captured",
        "payment_intent.succeeded",
        "payment_intent.canceled",
        "charge.refunded",
        "refund.created",
        "customer.subscription.created",
        "invoice.created",
        "invoice.finalized",
        "invoice.paid",
        "customer.subscription.updated",
    ]

    def order(e: dict) -> tuple:
        pos = lifecycle.index(e["type"]) if e["type"] in lifecycle else len(lifecycle)
        return (e["created"], pos, e["id"])

    return sorted(related, key=order)


def _guarded(flow_key: str, title: str, body: Callable[[_Run], None]) -> SandboxTrace:
    """Run ``body`` with the standard safety net: key errors and unexpected
    Stripe errors become a message on the trace, never an exception."""
    started = time.perf_counter()
    try:
        run = _Run(flow_key, title)
    except SandboxKeyError as error:
        trace = SandboxTrace(flow=flow_key, title=title, error=str(error))
        trace.duration_ms = round((time.perf_counter() - started) * 1000)
        return trace
    try:
        body(run)
        run.collect_events()
    except stripe.StripeError as error:
        run.trace.error = (
            f"Stripe API error: {getattr(error, 'user_message', None) or error}"
        )
    run.trace.duration_ms = round((time.perf_counter() - started) * 1000)
    return run.trace


# --- HTTP wire form --------------------------------------------------------
# A trace step records an SDK-style call name ("PaymentIntent.create") plus
# the exact params sent. The UI shows each step as the HTTP request it
# becomes — a runnable curl command — so any step can be copied into a
# terminal and replayed against the reader's own test key.

# call name -> (verb, path template, request key whose value goes in the path).
_ENDPOINTS: dict[str, tuple[str, str, str | None]] = {
    "PaymentIntent.create": ("POST", "/v1/payment_intents", None),
    "PaymentIntent.capture": ("POST", "/v1/payment_intents/{id}/capture", "payment_intent"),
    "PaymentIntent.cancel": ("POST", "/v1/payment_intents/{id}/cancel", "payment_intent"),
    "Refund.create": ("POST", "/v1/refunds", None),
    "Customer.create": ("POST", "/v1/customers", None),
    "Customer.update": ("POST", "/v1/customers/{id}", "customer"),
    "PaymentMethod.attach": ("POST", "/v1/payment_methods/{id}/attach", "payment_method"),
    "Price.create": ("POST", "/v1/prices", None),
    "Subscription.create": ("POST", "/v1/subscriptions", None),
    "Subscription.update": ("POST", "/v1/subscriptions/{id}", "subscription"),
    "SetupIntent.create": ("POST", "/v1/setup_intents", None),
    "checkout.Session.create": ("POST", "/v1/checkout/sessions", None),
    "PaymentLink.create": ("POST", "/v1/payment_links", None),
    "Dispute.retrieve": ("GET", "/v1/disputes/{id}", "dispute"),
    "Dispute.update": ("POST", "/v1/disputes/{id}", "dispute"),
}

_PLACEHOLDER = re.compile(r"^<.+>$")  # preview stand-ins like "<customer id>"


def http_call(call: str, request: dict) -> tuple[str, str]:
    """('POST', '/v1/payment_intents/pi_123/capture') for a recorded step."""
    verb, path, id_key = _ENDPOINTS.get(call, ("POST", f"/v1/{call}", None))
    if id_key:
        path = path.replace("{id}", str(request.get(id_key, "{id}")))
    return verb, path


def _form_fields(value: Any, prefix: str = ""):
    """Flatten a params dict into Stripe's form encoding:
    {"items": [{"price": "p"}]} -> [("items[0][price]", "p")]."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _form_fields(inner, f"{prefix}[{key}]" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for i, inner in enumerate(value):
            yield from _form_fields(inner, f"{prefix}[{i}]")
    elif isinstance(value, bool):
        yield prefix, "true" if value else "false"
    else:
        yield prefix, str(value)


_PLAIN = re.compile(r"^[A-Za-z0-9_.:/@-]+$")


def curl_command(call: str, request: dict) -> str:
    """The step as a runnable curl command. Preview placeholders
    ("<customer id>") pass through untouched, flagged by a comment line."""
    verb, path = http_call(call, request)
    id_key = _ENDPOINTS.get(call, ("", "", None))[2]
    body = {k: v for k, v in request.items() if k != id_key}
    fields = list(_form_fields(body))
    lines = []
    if any(_PLACEHOLDER.match(v) for _, v in fields) or "<" in path:
        lines.append("# values in <angle brackets> come from an earlier step's response")
    # -G moves the -d fields into the query string; a bare GET needs nothing.
    flag = "-G " if verb == "GET" and fields else ""
    lines.append(f"curl {flag}https://api.stripe.com{path} \\")
    auth = '  -u "$STRIPE_SECRET_KEY:"'
    if not fields:
        if verb == "GET":
            lines.append(auth)
            return "\n".join(lines)
        # curl needs the verb spelled out when there is no -d to imply POST.
        lines.append(auth + " \\")
        lines.append("  -X POST")
        return "\n".join(lines)
    lines.append(auth + " \\")
    for i, (key, value) in enumerate(fields):
        pair = f"{key}={value}"
        if not (_PLAIN.match(key) and _PLAIN.match(str(value))):
            pair = f'"{pair}"'
        tail = " \\" if i < len(fields) - 1 else ""
        lines.append(f"  -d {pair}{tail}")
    return "\n".join(lines)


# --- Shared request builders ----------------------------------------------
# Both the preview and the execution call these, so the payload a user sees
# before clicking is by construction the payload that gets sent.


def _payment_request(amount_cents: int, **extra) -> dict:
    return {
        "amount": amount_cents,
        "currency": CURRENCY,
        "confirm": True,
        "payment_method": TEST_CARD,
        "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
        **extra,
    }


def _recurring_price_request(amount_cents: int, name: str) -> dict:
    return {
        "currency": CURRENCY,
        "unit_amount": amount_cents,
        "recurring": {"interval": "month"},
        "product_data": {"name": name},
    }


def _subscribe_customer(
    run: _Run, monthly_cents: int, plan_name: str, trial_days: int | None = None
):
    """Create a price, a customer with a saved test card, and subscribe
    them. Shared by the subscription flows; ``trial_days`` delays the first
    charge behind a free trial."""
    price_req = _recurring_price_request(monthly_cents, plan_name)
    price = run.record(
        "Price.create", price_req, run.client.v1.prices.create(params=price_req),
        note=f"{_money(monthly_cents)}/month recurring price",
    )
    cust_req = {"description": "Sandbox demo customer"}
    customer = run.record(
        "Customer.create", cust_req, run.client.v1.customers.create(params=cust_req),
        note="the subscriber",
    )
    attach_req = {"payment_method": TEST_CARD, "customer": customer.id}
    method = run.record(
        "PaymentMethod.attach", attach_req,
        run.client.v1.payment_methods.attach(TEST_CARD, params={"customer": customer.id}),
        note="save the test card on the customer",
    )
    default_req = {"invoice_settings": {"default_payment_method": method.id}}
    run.record(
        "Customer.update", {"customer": customer.id, **default_req},
        run.client.v1.customers.update(customer.id, params=default_req),
        note="make it the default for invoices",
    )
    sub_req: dict = {"customer": customer.id, "items": [{"price": price.id}]}
    if trial_days:
        sub_req["trial_period_days"] = trial_days
    subscription = run.record(
        "Subscription.create", sub_req,
        run.client.v1.subscriptions.create(params=sub_req),
        note=(
            f"free for {trial_days} days — the first charge comes when the "
            "trial ends"
            if trial_days
            else "first invoice is charged immediately"
        ),
    )
    return subscription


def _subscribe_plan_steps(
    monthly_cents: int, plan_name: str, trial_days: int | None = None
) -> list[dict]:
    """Preview counterpart to ``_subscribe_customer``."""
    return [
        {
            "call": "Price.create",
            "request": _recurring_price_request(monthly_cents, plan_name),
            "note": f"{_money(monthly_cents)}/month recurring price",
        },
        {
            "call": "Customer.create",
            "request": {"description": "Sandbox demo customer"},
            "note": "the subscriber",
        },
        {
            "call": "PaymentMethod.attach",
            "request": {"payment_method": TEST_CARD, "customer": "<customer id>"},
            "note": "save the test card on the customer",
        },
        {
            "call": "Customer.update",
            "request": {
                "customer": "<customer id>",
                "invoice_settings": {"default_payment_method": "<payment method id>"},
            },
            "note": "make it the default for invoices",
        },
        {
            "call": "Subscription.create",
            "request": {
                "customer": "<customer id>",
                "items": [{"price": "<price id>"}],
                **({"trial_period_days": trial_days} if trial_days else {}),
            },
            "note": (
                f"free for {trial_days} days — the first charge comes when "
                "the trial ends"
                if trial_days
                else "first invoice is charged immediately"
            ),
        },
    ]


# --- Flow: payment and refund ---------------------------------------------


def plan_payment_and_refund(amount_cents: int, refund_amount_cents: int) -> list[dict]:
    partial = 0 < refund_amount_cents < amount_cents
    refund_req: dict = {"payment_intent": "<id returned by step 1>"}
    if partial:
        refund_req["amount"] = refund_amount_cents
    return [
        {
            "call": "PaymentIntent.create",
            "request": _payment_request(amount_cents),
            "note": f"charge {_money(amount_cents)} to a test card",
        },
        {
            "call": "Refund.create",
            "request": refund_req,
            "note": (
                f"refund {_money(refund_amount_cents)} of it"
                if partial
                else "refund the full amount"
            ),
        },
    ]


def run_payment_and_refund(amount_cents: int = 2000, refund_amount_cents: int = 0) -> SandboxTrace:
    partial = 0 < refund_amount_cents < amount_cents
    title = (
        f"Charge {_money(amount_cents)}, then refund "
        + (_money(refund_amount_cents) if partial else "it in full")
    )

    def body(run: _Run):
        pay_req = _payment_request(amount_cents)
        intent = run.record(
            "PaymentIntent.create", pay_req,
            run.client.v1.payment_intents.create(params=pay_req),
            note="the customer is charged",
        )
        refund_req: dict = {"payment_intent": intent.id}
        if partial:
            refund_req["amount"] = refund_amount_cents
        run.record(
            "Refund.create", refund_req,
            run.client.v1.refunds.create(params=refund_req),
            note="the money goes back to the card",
        )

    return _guarded("payment_and_refund", title, body)


# --- Flow: charge a card ---------------------------------------------------


def plan_charge_card(amount_cents: int) -> list[dict]:
    return [
        {
            "call": "PaymentIntent.create",
            "request": _payment_request(amount_cents),
            "note": f"charge {_money(amount_cents)} to a test card",
        }
    ]


def run_charge_card(amount_cents: int = 2000) -> SandboxTrace:
    title = f"Charge {_money(amount_cents)} to a test card"

    def body(run: _Run):
        req = _payment_request(amount_cents)
        run.record(
            "PaymentIntent.create", req,
            run.client.v1.payment_intents.create(params=req),
            note="the customer is charged",
        )

    return _guarded("charge_card", title, body)


# --- Flow: hold and capture ------------------------------------------------


def plan_hold_and_capture(
    amount_cents: int, capture_amount_cents: int = 0
) -> list[dict]:
    partial = 0 < capture_amount_cents < amount_cents
    capture_req: dict = {"payment_intent": "<id returned by step 1>"}
    if partial:
        capture_req["amount_to_capture"] = capture_amount_cents
    return [
        {
            "call": "PaymentIntent.create",
            "request": _payment_request(amount_cents, capture_method="manual"),
            "note": "places the hold on a test card",
        },
        {
            "call": "PaymentIntent.capture",
            "request": capture_req,
            "note": (
                f"captures {_money(capture_amount_cents)} — the rest of the "
                "hold is released"
                if partial
                else "captures the held funds"
            ),
        },
    ]


def run_hold_and_capture(
    amount_cents: int = 5000, capture_amount_cents: int = 0
) -> SandboxTrace:
    partial = 0 < capture_amount_cents < amount_cents
    title = f"Hold {_money(amount_cents)} on a test card, then capture " + (
        _money(capture_amount_cents) if partial else "it"
    )

    def body(run: _Run):
        create_req = _payment_request(amount_cents, capture_method="manual")
        intent = run.record(
            "PaymentIntent.create", create_req,
            run.client.v1.payment_intents.create(params=create_req),
            note="money is held on the card, not yet moved",
        )
        capture_req: dict = {"payment_intent": intent.id}
        capture_params: dict = {}
        if partial:
            capture_req["amount_to_capture"] = capture_amount_cents
            capture_params["amount_to_capture"] = capture_amount_cents
        run.record(
            "PaymentIntent.capture", capture_req,
            run.client.v1.payment_intents.capture(intent.id, params=capture_params),
            note=(
                "the captured part moves; the rest of the hold is released"
                if partial
                else "the held funds are now captured"
            ),
        )

    return _guarded("hold_and_capture", title, body)


# --- Flow: hold and release ------------------------------------------------


def plan_hold_and_release(amount_cents: int) -> list[dict]:
    return [
        {
            "call": "PaymentIntent.create",
            "request": _payment_request(amount_cents, capture_method="manual"),
            "note": "places the hold on a test card",
        },
        {
            "call": "PaymentIntent.cancel",
            "request": {"payment_intent": "<id returned by step 1>"},
            "note": "releases the hold — no refund needed, no money moved",
        },
    ]


def run_hold_and_release(amount_cents: int = 5000) -> SandboxTrace:
    title = f"Hold {_money(amount_cents)} on a test card, then release it"

    def body(run: _Run):
        create_req = _payment_request(amount_cents, capture_method="manual")
        intent = run.record(
            "PaymentIntent.create", create_req,
            run.client.v1.payment_intents.create(params=create_req),
            note="money is held on the card, not yet moved",
        )
        run.record(
            "PaymentIntent.cancel", {"payment_intent": intent.id},
            run.client.v1.payment_intents.cancel(intent.id),
            note="the hold is released without ever charging the customer",
        )

    return _guarded("hold_and_release", title, body)


# --- Flow: subscription lifecycle -----------------------------------------


def plan_subscription_lifecycle(monthly_amount_cents: int) -> list[dict]:
    return _subscribe_plan_steps(monthly_amount_cents, "Sandbox demo plan") + [
        {
            "call": "Subscription.update",
            "request": {"subscription": "<subscription id>", "cancel_at_period_end": True},
            "note": "cancel at period end — stays active until the paid period runs out",
        }
    ]


def run_subscription_lifecycle(monthly_amount_cents: int = 1200) -> SandboxTrace:
    title = (
        f"Subscribe a customer at {_money(monthly_amount_cents)}/month, "
        "then cancel at period end"
    )

    def body(run: _Run):
        subscription = _subscribe_customer(run, monthly_amount_cents, "Sandbox demo plan")
        cancel_req = {"cancel_at_period_end": True}
        run.record(
            "Subscription.update", {"subscription": subscription.id, **cancel_req},
            run.client.v1.subscriptions.update(subscription.id, params=cancel_req),
            note="still active, but it will not renew",
        )

    return _guarded("subscription_lifecycle", title, body)


# --- Flow: subscription price change --------------------------------------


def plan_subscription_change_price(
    monthly_amount_cents: int, new_monthly_amount_cents: int
) -> list[dict]:
    return _subscribe_plan_steps(monthly_amount_cents, "Sandbox demo plan") + [
        {
            "call": "Price.create",
            "request": _recurring_price_request(new_monthly_amount_cents, "Sandbox demo plan (new tier)"),
            "note": f"the new {_money(new_monthly_amount_cents)}/month price",
        },
        {
            "call": "Subscription.update",
            "request": {
                "subscription": "<subscription id>",
                "items": [{"id": "<subscription item id>", "price": "<new price id>"}],
                "proration_behavior": "create_prorations",
            },
            "note": "swap the price and prorate the difference",
        },
    ]


def run_subscription_change_price(
    monthly_amount_cents: int = 1200, new_monthly_amount_cents: int = 3000
) -> SandboxTrace:
    title = (
        f"Move a subscriber from {_money(monthly_amount_cents)} to "
        f"{_money(new_monthly_amount_cents)}/month with proration"
    )

    def body(run: _Run):
        subscription = _subscribe_customer(run, monthly_amount_cents, "Sandbox demo plan")
        price_req = _recurring_price_request(
            new_monthly_amount_cents, "Sandbox demo plan (new tier)"
        )
        new_price = run.record(
            "Price.create", price_req, run.client.v1.prices.create(params=price_req),
            note="the tier the customer is moving to",
        )
        item_id = subscription["items"].data[0].id
        update_req = {
            "items": [{"id": item_id, "price": new_price.id}],
            "proration_behavior": "create_prorations",
        }
        run.record(
            "Subscription.update", {"subscription": subscription.id, **update_req},
            run.client.v1.subscriptions.update(subscription.id, params=update_req),
            note="Stripe credits the unused time and bills the difference",
        )

    return _guarded("subscription_change_price", title, body)


# --- Flow: subscription with a free trial ----------------------------------


def plan_subscription_trial(monthly_amount_cents: int, trial_days: int) -> list[dict]:
    return _subscribe_plan_steps(monthly_amount_cents, "Sandbox demo plan", trial_days)


def run_subscription_trial(
    monthly_amount_cents: int = 1200, trial_days: int = 14
) -> SandboxTrace:
    title = (
        f"Start a {_money(monthly_amount_cents)}/month subscription with a "
        f"{trial_days}-day free trial"
    )

    def body(run: _Run):
        _subscribe_customer(run, monthly_amount_cents, "Sandbox demo plan", trial_days)

    return _guarded("subscription_trial", title, body)


# --- Flow: checkout session -----------------------------------------------


def _checkout_request(amount_cents: int) -> dict:
    return {
        "mode": "payment",
        "success_url": "https://example.com/success",
        "cancel_url": "https://example.com/cancel",
        "line_items": [
            {
                "price_data": {
                    "currency": CURRENCY,
                    "product_data": {"name": "Sandbox demo item"},
                    "unit_amount": amount_cents,
                },
                "quantity": 1,
            }
        ],
    }


def plan_checkout_session(amount_cents: int) -> list[dict]:
    return [
        {
            "call": "checkout.Session.create",
            "request": _checkout_request(amount_cents),
            "note": "returns a hosted payment page you can open and pay with a test card",
        }
    ]


def run_checkout_session(amount_cents: int = 2500) -> SandboxTrace:
    title = f"Create a hosted Checkout page for {_money(amount_cents)}"

    def body(run: _Run):
        req = _checkout_request(amount_cents)
        session = run.client.v1.checkout.sessions.create(params=req)
        run.record(
            "checkout.Session.create", req, session,
            note="open the link and pay with card 4242 4242 4242 4242",
            link=getattr(session, "url", None),
        )

    return _guarded("checkout_session", title, body)


# --- Flow: payment link ----------------------------------------------------


def _one_time_price_request(amount_cents: int) -> dict:
    return {
        "currency": CURRENCY,
        "unit_amount": amount_cents,
        "product_data": {"name": "Sandbox demo item"},
    }


def plan_payment_link(amount_cents: int) -> list[dict]:
    return [
        {
            "call": "Price.create",
            "request": _one_time_price_request(amount_cents),
            "note": f"a one-time {_money(amount_cents)} price for the link to sell",
        },
        {
            "call": "PaymentLink.create",
            "request": {"line_items": [{"price": "<price id>", "quantity": 1}]},
            "note": "returns a reusable payment page URL you can share anywhere",
        },
    ]


def run_payment_link(amount_cents: int = 2500) -> SandboxTrace:
    title = f"Create a shareable Payment Link for {_money(amount_cents)}"

    def body(run: _Run):
        price_req = _one_time_price_request(amount_cents)
        price = run.record(
            "Price.create", price_req,
            run.client.v1.prices.create(params=price_req),
            note="the product the link sells",
        )
        link_req = {"line_items": [{"price": price.id, "quantity": 1}]}
        link = run.client.v1.payment_links.create(params=link_req)
        run.record(
            "PaymentLink.create", link_req, link,
            note="share this URL anywhere — no code on your site",
            link=getattr(link, "url", None),
        )

    return _guarded("payment_link", title, body)


# --- Flow: save a card and charge it later --------------------------------


def _setup_intent_request(customer_id: str) -> dict:
    return {
        "customer": customer_id,
        "confirm": True,
        "payment_method": TEST_CARD,
        "usage": "off_session",
        "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
    }


def plan_save_card_off_session(amount_cents: int) -> list[dict]:
    return [
        {
            "call": "Customer.create",
            "request": {"description": "Sandbox demo customer"},
            "note": "who the card is saved against",
        },
        {
            "call": "SetupIntent.create",
            "request": _setup_intent_request("<customer id>"),
            "note": "save the card for future use without charging it",
        },
        {
            "call": "PaymentIntent.create",
            "request": {
                "amount": amount_cents,
                "currency": CURRENCY,
                "customer": "<customer id>",
                "payment_method": "<saved payment method id>",
                "off_session": True,
                "confirm": True,
            },
            "note": f"charge {_money(amount_cents)} later, customer not present",
        },
    ]


def run_save_card_off_session(amount_cents: int = 1500) -> SandboxTrace:
    title = f"Save a card, then charge it {_money(amount_cents)} off-session"

    def body(run: _Run):
        cust_req = {"description": "Sandbox demo customer"}
        customer = run.record(
            "Customer.create", cust_req, run.client.v1.customers.create(params=cust_req),
            note="who the card is saved against",
        )
        setup_req = _setup_intent_request(customer.id)
        setup = run.record(
            "SetupIntent.create", setup_req,
            run.client.v1.setup_intents.create(params=setup_req),
            note="card saved — no money moved",
        )
        charge_req = {
            "amount": amount_cents,
            "currency": CURRENCY,
            "customer": customer.id,
            "payment_method": setup.payment_method,
            "off_session": True,
            "confirm": True,
        }
        run.record(
            "PaymentIntent.create", charge_req,
            run.client.v1.payment_intents.create(params=charge_req),
            note="charged with the customer nowhere near the checkout",
        )

    return _guarded("save_card_off_session", title, body)


# --- Flow: declined card ---------------------------------------------------

DECLINE_CARDS = {
    "generic": (
        "pm_card_chargeDeclined",
        "a card the bank rejects outright",
    ),
    "insufficient_funds": (
        "pm_card_chargeDeclinedInsufficientFunds",
        "a card with no money behind it",
    ),
    "authentication_required": (
        "pm_card_authenticationRequired",
        "a card that demands 3D Secure before it will pay",
    ),
}


def plan_declined_card(decline_type: str) -> list[dict]:
    card, description = DECLINE_CARDS[decline_type]
    return [
        {
            "call": "PaymentIntent.create",
            "request": _payment_request(2000) | {"payment_method": card},
            "note": f"attempt a payment with {description}",
        }
    ]


def run_declined_card(decline_type: str = "generic") -> SandboxTrace:
    card, description = DECLINE_CARDS[decline_type]
    title = f"Attempt a payment with {description}"

    def body(run: _Run):
        req = _payment_request(2000) | {"payment_method": card}
        try:
            run.record(
                "PaymentIntent.create", req,
                run.client.v1.payment_intents.create(params=req),
                note=(
                    "not paid — the payment stops and waits for the customer "
                    "to authenticate"
                    if decline_type == "authentication_required"
                    else "unexpectedly succeeded"
                ),
            )
        except stripe.CardError as error:
            run.record_failure(
                "PaymentIntent.create", req, error,
                note="this is the error your integration must handle",
            )

    return _guarded("declined_card", title, body)


# --- Flow: dispute ---------------------------------------------------------


def plan_dispute(amount_cents: int) -> list[dict]:
    return [
        {
            "call": "PaymentIntent.create",
            "request": _payment_request(amount_cents, payment_method=DISPUTE_CARD),
            "note": "charge a test card that always triggers a dispute",
        },
        {
            "call": "Dispute.retrieve",
            "request": {"dispute": "<dispute id — Stripe opens it moments after the charge>"},
            "note": "read the dispute the cardholder's bank opened",
        },
        {
            "call": "Dispute.update",
            "request": {
                "dispute": "<dispute id>",
                "evidence": {"uncategorized_text": "winning_evidence"},
                "submit": True,
            },
            "note": "respond with evidence — in test mode this exact text wins",
        },
    ]


def run_dispute(amount_cents: int = 2000) -> SandboxTrace:
    title = (
        f"Charge {_money(amount_cents)}, watch it get disputed, then "
        "respond with evidence"
    )

    def body(run: _Run):
        pay_req = _payment_request(amount_cents, payment_method=DISPUTE_CARD)
        intent = run.record(
            "PaymentIntent.create", pay_req,
            run.client.v1.payment_intents.create(params=pay_req),
            note="the charge succeeds — the dispute lands moments later",
        )
        # The dispute is created asynchronously, almost always within a
        # second or two — poll briefly instead of guessing a sleep.
        dispute = None
        for _ in range(15):
            found = run.client.v1.disputes.list(
                params={"payment_intent": intent.id, "limit": 1}
            )
            if found.data:
                dispute = found.data[0]
                break
            time.sleep(1)
        if dispute is None:
            run.trace.error = (
                "Stripe had not opened the test dispute after 15 seconds — "
                "run the flow again."
            )
            return
        dispute = run.record(
            "Dispute.retrieve", {"dispute": dispute.id},
            run.client.v1.disputes.retrieve(dispute.id),
            note="the disputed money and a dispute fee are on hold",
        )
        update_params = {
            "evidence": {"uncategorized_text": "winning_evidence"},
            "submit": True,
        }
        run.record(
            "Dispute.update", {"dispute": dispute.id, **update_params},
            run.client.v1.disputes.update(dispute.id, params=update_params),
            note="in test mode this evidence text wins the dispute",
        )

    return _guarded("dispute", title, body)


# --- Flow registry ---------------------------------------------------------


@dataclass(frozen=True)
class Flow:
    key: str
    label: str  # shown in the UI
    description: str  # shown to the router, which picks between flows
    params: dict[str, ParamSpec]
    plan: Callable[..., list[dict]]
    run: Callable[..., SandboxTrace]
    pattern: re.Pattern  # fallback matcher, used only when the router fails


AMOUNT = ParamSpec("Amount (cents)", 2000)

FLOWS: dict[str, Flow] = {
    "payment_and_refund": Flow(
        key="payment_and_refund",
        label="charge a card, then refund it",
        description="Charging a card and refunding it, in full or partially.",
        params={
            "amount_cents": ParamSpec("Charge amount (cents)", 2000),
            "refund_amount_cents": ParamSpec(
                "Refund amount (cents, 0 = full refund)", 0, minimum=0
            ),
        },
        plan=plan_payment_and_refund,
        run=run_payment_and_refund,
        pattern=re.compile(r"refund|money back|reimburse", re.IGNORECASE),
    ),
    "hold_and_capture": Flow(
        key="hold_and_capture",
        label="place a hold, then capture it",
        description=(
            "Authorising a payment now and capturing the money later, e.g. "
            "charging when an order ships."
        ),
        params={
            "amount_cents": ParamSpec("Hold amount (cents)", 5000),
            "capture_amount_cents": ParamSpec(
                "Capture amount (cents, 0 = full)", 0, minimum=0
            ),
        },
        plan=plan_hold_and_capture,
        run=run_hold_and_capture,
        pattern=re.compile(
            r"\bhold\b|captur|authoriz|charge.{0,40}(later|when|after|ship)|uncaptured",
            re.IGNORECASE,
        ),
    ),
    "hold_and_release": Flow(
        key="hold_and_release",
        label="place a hold, then release it",
        description=(
            "Cancelling an authorisation instead of capturing it — releasing "
            "a hold without charging, and why that is not a refund."
        ),
        params={"amount_cents": ParamSpec("Hold amount (cents)", 5000)},
        plan=plan_hold_and_release,
        run=run_hold_and_release,
        pattern=re.compile(r"(cancel|release|void).{0,30}(hold|authoriz)", re.IGNORECASE),
    ),
    # Before the other subscription flows: "trial" is the more specific
    # keyword, and the fallback scan takes the first pattern that matches.
    "subscription_trial": Flow(
        key="subscription_trial",
        label="start a subscription with a free trial",
        description=(
            "Starting a subscription with a free trial — the card is saved "
            "but nothing is charged today; billing starts when the trial ends."
        ),
        params={
            "monthly_amount_cents": ParamSpec("Monthly price (cents)", 1200),
            "trial_days": ParamSpec("Trial length (days)", 14, minimum=1, maximum=365),
        },
        plan=plan_subscription_trial,
        run=run_subscription_trial,
        pattern=re.compile(r"free trial|trial", re.IGNORECASE),
    ),
    "subscription_lifecycle": Flow(
        key="subscription_lifecycle",
        label="subscribe a customer, then cancel at period end",
        description=(
            "Creating a recurring subscription for a customer and cancelling "
            "it, including cancel-at-period-end behaviour."
        ),
        params={"monthly_amount_cents": ParamSpec("Monthly price (cents)", 1200)},
        plan=plan_subscription_lifecycle,
        run=run_subscription_lifecycle,
        pattern=re.compile(r"subscri|recurring|billing cycle|cancel.{0,20}plan", re.IGNORECASE),
    ),
    "subscription_change_price": Flow(
        key="subscription_change_price",
        label="move a subscriber to a new price, with proration",
        description=(
            "Upgrading or downgrading an existing subscription to a different "
            "price, and how proration credits the unused time."
        ),
        params={
            "monthly_amount_cents": ParamSpec("Current monthly price (cents)", 1200),
            "new_monthly_amount_cents": ParamSpec("New monthly price (cents)", 3000),
        },
        plan=plan_subscription_change_price,
        run=run_subscription_change_price,
        pattern=re.compile(r"prorat|upgrade|downgrade|change.{0,20}(price|plan)", re.IGNORECASE),
    ),
    "checkout_session": Flow(
        key="checkout_session",
        label="create a hosted Checkout page",
        description=(
            "Creating a Stripe Checkout Session — the hosted payment page — "
            "and getting the URL to send the customer to."
        ),
        params={"amount_cents": ParamSpec("Item price (cents)", 2500)},
        plan=plan_checkout_session,
        run=run_checkout_session,
        pattern=re.compile(r"checkout session|checkout\.session|hosted page|payment page", re.IGNORECASE),
    ),
    "payment_link": Flow(
        key="payment_link",
        label="create a shareable Payment Link",
        description=(
            "Creating a Payment Link — a reusable hosted payment page URL "
            "you can share anywhere, with no code on your site."
        ),
        params={"amount_cents": ParamSpec("Item price (cents)", 2500)},
        plan=plan_payment_link,
        run=run_payment_link,
        pattern=re.compile(r"payment.?link", re.IGNORECASE),
    ),
    "save_card_off_session": Flow(
        key="save_card_off_session",
        label="save a card, then charge it off-session",
        description=(
            "Saving a customer's card with a SetupIntent and charging it "
            "later when the customer is not present."
        ),
        params={"amount_cents": ParamSpec("Later charge (cents)", 1500)},
        plan=plan_save_card_off_session,
        run=run_save_card_off_session,
        pattern=re.compile(
            r"setup.?intent|save.{0,15}card|off.?session|reuse.{0,15}card|future payment",
            re.IGNORECASE,
        ),
    ),
    "declined_card": Flow(
        key="declined_card",
        label="watch a payment get declined",
        description=(
            "What a failed payment looks like: declines, insufficient funds, "
            "and cards that require 3D Secure authentication."
        ),
        params={
            "decline_type": ParamSpec(
                "Decline to simulate",
                "generic",
                kind="choice",
                choices=tuple(DECLINE_CARDS),
            )
        },
        plan=plan_declined_card,
        run=run_declined_card,
        pattern=re.compile(
            r"declin|insufficient funds|card.{0,15}fail|3d.?secure|\b3ds\b|authentication required",
            re.IGNORECASE,
        ),
    ),
    "dispute": Flow(
        key="dispute",
        label="get a dispute, then respond with evidence",
        description=(
            "What a dispute (chargeback) looks like: a charge is disputed by "
            "the cardholder's bank, and evidence is submitted in response."
        ),
        params={"amount_cents": ParamSpec("Charge amount (cents)", 2000)},
        plan=plan_dispute,
        run=run_dispute,
        pattern=re.compile(r"disput|chargeback", re.IGNORECASE),
    ),
    # Deliberately last: its pattern is the broadest, and the keyword
    # fallback scans this dict in order — every more specific flow above
    # gets first claim on a question before "just charge a card" does.
    "charge_card": Flow(
        key="charge_card",
        label="charge a card",
        description=(
            "Charging a customer's card once — a plain one-time payment, "
            "with no refund, hold, or follow-up step."
        ),
        params={"amount_cents": ParamSpec("Charge amount (cents)", 2000)},
        plan=plan_charge_card,
        run=run_charge_card,
        pattern=re.compile(
            r"charg|credit card|debit card|\bpay\b|one.?time payment|accept.{0,20}payment",
            re.IGNORECASE,
        ),
    ),
}


def flow_catalog() -> list[dict]:
    """The whitelist as the router sees it: what each flow demonstrates and
    which parameters it accepts."""
    return [
        {
            "key": flow.key,
            "description": flow.description,
            "params": {
                name: {
                    "label": spec.label,
                    "default": spec.default,
                    # Choice parameters are useless to the router unless it
                    # is told what the choices are.
                    "choices": list(spec.choices),
                }
                for name, spec in flow.params.items()
            },
        }
        for flow in FLOWS.values()
    ]


def match_flow(question: str) -> str | None:
    """Keyword fallback used only when the router could not choose — a
    router outage should not silently remove the sandbox."""
    for key, flow in FLOWS.items():
        if flow.pattern.search(question):
            return key
    return None


def preview_flow(key: str, params: dict | None = None) -> list[dict]:
    """The flow's planned calls and payloads, without executing anything."""
    return FLOWS[key].plan(**coerce_params(key, params))


def run_flow(key: str, params: dict | None = None) -> SandboxTrace:
    return FLOWS[key].run(**coerce_params(key, params))


if __name__ == "__main__":
    import sys

    flow_key = sys.argv[1] if len(sys.argv) > 1 else "hold_and_capture"
    if flow_key not in FLOWS:
        raise SystemExit(f"Unknown flow. Available: {', '.join(FLOWS)}")
    result = run_flow(flow_key)
    if result.error:
        raise SystemExit(result.error)
    print(result.title)
    for i, step in enumerate(result.steps, start=1):
        mark = "x" if step.failed else "-"
        print(f"{mark} {i}. {step.call} -> {step.object_id} [{step.status}]  ({step.note})")
    print("events:", " -> ".join(e["type"] for e in result.events) or "(none yet)")
    print(f"({result.duration_ms} ms)")
