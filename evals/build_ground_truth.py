"""Build evals/ground_truth.jsonl from curated Stack Overflow candidates.

Curation happened by hand (2026-08-17): each kept candidate was mapped to the
single corpus page that canonically answers it. Candidates whose answer lives
on a page outside the corpus (test clocks, customer search, appearance API,
...) were skipped — with exact-URL scoring they could only produce false
misses. Synthetic questions were written *from* corpus pages, so their
mapping is correct by construction, and they cover product areas the SO set
doesn't reach (disputes, payouts, radar, terminal, identity).

Run: uv run python -m evals.build_ground_truth
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "data" / "so_question_candidates.jsonl"
CORPUS_URLS = ROOT / "data" / "corpus_page_urls.txt"
OUT = ROOT / "evals" / "ground_truth.jsonl"

DOCS = "https://docs.stripe.com"

# candidate line number (1-based) -> answer page path in the corpus
SO_MAPPING: dict[int, str] = {
    1: "/webhooks/signature.md",
    5: "/payments/payment-element/control-billing-details-collection.md",
    6: "/testing.md",
    7: "/api/errors.md",
    9: "/currencies.md",
    19: "/api/checkout/sessions/create.md",
    20: "/billing/subscriptions/change-price.md",
    22: "/checkout/fulfillment.md",
    32: "/api/customers.md",
    34: "/payments/setup-intents.md",
    36: "/payments/collect-addresses.md",
    37: "/tax/checkout.md",
    38: "/api/payment_methods/object.md",
    39: "/connect/authentication.md",
    41: "/payments/save-and-reuse.md",
    42: "/connect/charges.md",
    44: "/event-destinations.md",
    45: "/billing/subscriptions/cancel.md",
    47: "/api/events/types.md",
    52: "/api/subscriptions.md",
    53: "/testing/wallets.md",
    57: "/webhooks.md",
    58: "/api/payment_methods.md",
    59: "/connect/payouts-connected-accounts.md",
    60: "/billing/subscriptions/billing-cycle.md",
    61: "/payments/payment-methods.md",
    62: "/payments/checkout/discounts.md",
    64: "/upgrades.md",
    67: "/billing/subscriptions/change.md",
    76: "/payments/accept-a-payment.md",
    77: "/payments/payment-intents.md",
    78: "/payments/save-during-payment.md",
    81: "/security/guide.md",
    82: "/payments/place-a-hold-on-a-payment-method.md",
    87: "/payments/payment-methods/integration-options.md",
    98: "/billing/subscriptions/coupons.md",
    99: "/billing/subscriptions/billing-cycle.md",
    100: "/payments/payment-intents/verifying-status.md",
    103: "/api/checkout/sessions.md",
    107: "/billing/customer.md",
    114: "/api/payment_intents/create.md",
    117: "/payments/place-a-hold-on-a-payment-method.md",
    118: "/refunds.md",
}

# (question, answer page path) — written from the page, mapping is exact
SYNTHETIC: list[tuple[str, str]] = [
    (
        "What does the decline code insufficient_funds mean and what should I tell the customer?",
        "/declines/codes.md",
    ),
    (
        "How often does Stripe pay out my balance to my bank account?",
        "/payouts.md",
    ),
    (
        "How do I respond to a dispute programmatically through the API?",
        "/disputes/api.md",
    ),
    (
        "How do I write a custom Radar rule to block payments from certain countries?",
        "/radar/rules.md",
    ),
    (
        "How can I share a Stripe Payment Link with my customers?",
        "/payment-links/share.md",
    ),
    (
        "How do I pause payment collection on a subscription without cancelling it?",
        "/billing/subscriptions/pause-payment.md",
    ),
    (
        "How do I email an invoice to a customer with Stripe?",
        "/invoicing/send-email.md",
    ),
    (
        "How do I enable Stripe Tax so taxes are calculated automatically?",
        "/tax/set-up.md",
    ),
    (
        "How do I start accepting in-person payments with Stripe Terminal?",
        "/terminal/quickstart.md",
    ),
    (
        "How do I verify a user's government-issued ID with Stripe?",
        "/identity/verify-identity-documents.md",
    ),
    (
        "How do I accept Apple Pay and Google Pay on my website?",
        "/payments/wallets.md",
    ),
    (
        "What should I check before taking my Stripe integration live?",
        "/get-started/checklist/go-live.md",
    ),
]


def main() -> None:
    corpus = set(CORPUS_URLS.read_text().split())
    candidates = [
        json.loads(line)
        for line in CANDIDATES.read_text().splitlines()
        if line.strip()
    ]

    rows: list[dict] = []
    for lineno, path in sorted(SO_MAPPING.items()):
        cand = candidates[lineno - 1]
        rows.append(
            {
                "question": cand["question"],
                "source": "stackoverflow",
                "source_url": cand["source_url"],
                "answer_page_url": DOCS + path,
            }
        )
    for question, path in SYNTHETIC:
        rows.append(
            {
                "question": question,
                "source": "synthetic",
                "source_url": None,
                "answer_page_url": DOCS + path,
            }
        )

    missing = [r["answer_page_url"] for r in rows if r["answer_page_url"] not in corpus]
    if missing:
        raise SystemExit(f"answer pages not in corpus: {missing}")

    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions -> {OUT}")
    so = sum(1 for r in rows if r["source"] == "stackoverflow")
    print(f"  stackoverflow: {so}, synthetic: {len(rows) - so}")


if __name__ == "__main__":
    main()
