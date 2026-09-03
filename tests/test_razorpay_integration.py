"""The Razorpay recon adapter, and the boundary it is not allowed to cross.

Two things are being protected here.

First, the mapping itself: a settlement recon report has to arrive in the engine
as the same records the CSV path produces, with the fee-and-GST arithmetic
closing in integer paise. Money that drifts by a paisa between the gateway and
the matcher would show up as a tolerance question rather than as the bug it is.

Second, the network boundary. This adapter is the only part of the project that
can make an HTTP request. If it were ever importable from ``core/``, the claim
that matching is offline and deterministic would quietly stop being true - so
that is asserted here as well as in test_layer_separation.
"""

from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path

import pytest

from core.loader import load_batch
from core.matcher import reconcile
from integrations import razorpay

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "razorpay_sample" / "recon_combined.json"
SAMPLE_BANK = ROOT / "data" / "razorpay_sample" / "bank_statement.csv"


@pytest.fixture
def entities():
    return razorpay.load_fixture(FIXTURE)


# --------------------------------------------------------------------------
# Shape and mapping
# --------------------------------------------------------------------------

def test_fixture_matches_the_documented_recon_field_shape(entities):
    """Guards against the fixture drifting into a shape the real API never sends.

    A fixture that quietly grows a convenient field would make the adapter pass
    here and fail against Razorpay, which is the one failure this whole module
    exists to prevent.
    """
    documented = {
        "entity_id", "type", "debit", "credit", "amount", "currency", "fee",
        "tax", "on_hold", "settled", "created_at", "settled_at",
        "settlement_id", "utr", "order_id", "payment_id", "method",
        "description", "notes",
    }
    assert entities
    for e in entities:
        assert set(e) <= documented, f"fixture has undocumented fields: {set(e) - documented}"
        assert documented <= set(e), f"fixture is missing documented fields: {documented - set(e)}"


def test_amounts_are_integer_paise_end_to_end(entities):
    """No float ever touches money on the way in.

    The API hands over integer paise and the engine is integer-paise native, so
    the only place a rounding error could enter is this adapter. Asserting the
    identity rather than an approximate equality is the point.
    """
    for s in razorpay.settlements_from_recon(entities):
        assert isinstance(s.gross_paise, int)
        assert isinstance(s.net_paise, int)
        assert s.net_paise == s.gross_paise - s.fee_paise - s.tax_on_fee_paise


def test_payments_are_aggregated_into_the_payout_that_hits_the_bank(entities):
    """setl_RZA covers three payments and reaches the bank as one credit."""
    settlements = {s.transaction_id: s for s in razorpay.settlements_from_recon(entities)}
    a = settlements["setl_RZA"]
    contributing = [e for e in entities
                    if e["settlement_id"] == "setl_RZA" and e["type"] == "payment"]
    assert len(contributing) == 3
    assert a.gross_paise == sum(e["amount"] for e in contributing)
    assert a.utr_number == "327811004521"


def test_an_unsettled_payment_never_becomes_a_settlement(entities):
    """A payment Razorpay has not paid out yet is not a missing bank credit.

    Emitting it would manufacture an overdue receivable out of a payment that
    was never due, which is the exact confusion the cash layer separates.
    """
    ids = {s.transaction_id for s in razorpay.settlements_from_recon(entities)}
    unsettled = [e for e in entities if e.get("settlement_id") is None]
    assert unsettled, "fixture should carry at least one unsettled payment"
    for e in unsettled:
        assert e["payment_id"] not in ids
        assert e["entity_id"] not in ids


def test_a_refund_downgrades_its_order_status(entities):
    """The signal the reasoning layer reads when a credit falls short."""
    ledger = razorpay.ledger_from_recon(entities)
    refunded = {e["order_id"] for e in entities if e["type"] == "refund"}
    assert refunded
    for oid in refunded:
        assert ledger[oid].status == "partially_refunded"


def test_a_multi_order_settlement_carries_no_single_order_link(entities):
    """Attaching one arbitrary order to a payout covering three would be a lie
    in the audit trail, so the link is left empty instead."""
    settlements = {s.transaction_id: s for s in razorpay.settlements_from_recon(entities)}
    assert settlements["setl_RZA"].order_id is None
    # A single-order payout does keep its link.
    assert settlements["setl_RZB"].order_id == "order_RZ004"


def test_written_csvs_reload_without_losing_a_paisa(entities, tmp_path):
    razorpay.write_batch_csvs(entities, tmp_path)
    (tmp_path / "bank_statement.csv").write_text(
        SAMPLE_BANK.read_text(encoding="utf-8"), encoding="utf-8")

    direct = {s.transaction_id: s for s in razorpay.settlements_from_recon(entities)}
    reloaded = {s.transaction_id: s for s in load_batch(tmp_path).settlements}

    assert set(direct) == set(reloaded)
    for tid, s in direct.items():
        r = reloaded[tid]
        assert (r.gross_paise, r.fee_paise, r.tax_on_fee_paise, r.net_paise) == \
               (s.gross_paise, s.fee_paise, s.tax_on_fee_paise, s.net_paise)
        assert r.settlement_date == s.settlement_date
        assert r.utr_number == s.utr_number


def test_the_adapter_does_not_write_a_bank_statement(entities, tmp_path):
    """Razorpay cannot tell you what landed in your account. Writing an empty
    statement would reconcile everything to nothing and look like a clean run."""
    razorpay.write_batch_csvs(entities, tmp_path)
    assert (tmp_path / "settlement_report.csv").exists()
    assert (tmp_path / "internal_ledger.csv").exists()
    assert not (tmp_path / "bank_statement.csv").exists()


# --------------------------------------------------------------------------
# The engine, unmodified, on gateway-shaped data
# --------------------------------------------------------------------------

def test_the_engine_reconciles_a_razorpay_sourced_batch(entities, tmp_path):
    """The whole point: no branch in core for 'real' versus 'synthetic' input."""
    razorpay.write_batch_csvs(entities, tmp_path)
    (tmp_path / "bank_statement.csv").write_text(
        SAMPLE_BANK.read_text(encoding="utf-8"), encoding="utf-8")

    report = reconcile(load_batch(tmp_path))
    by_id = {r.record_id: r for r in report.results}

    # Clean UTR match.
    assert by_id["setl_RZA"].status == "matched"
    assert by_id["setl_RZA"].match_type == "exact_utr"

    # One payout, two credits: assembled by subset-sum, not by a one-to-one join.
    assert by_id["setl_RZC"].status == "matched_split"
    assert len(by_id["setl_RZC"].linked_ids["bank"]) == 2

    # UTR matches but the credit is short by the refund - held, not forced.
    assert by_id["setl_RZB"].status == "unresolved"
    assert by_id["setl_RZB"].exception_reason == "identifier_match_amount_discrepancy"

    # Settled by the gateway, nothing in the bank: a receivable, not a match.
    assert by_id["setl_RZD"].status == "unresolved"

    # An inflow belonging to no settlement is surfaced rather than absorbed.
    assert by_id["BNK_R0005"].exception_reason == "no_settlement_counterpart"


def test_the_refund_shortfall_is_exactly_the_refunded_amount(entities, tmp_path):
    """The exception is worth holding precisely because the gap is explainable.

    Asserting the arithmetic keeps the fixture honest: if the refund and the
    bank credit ever drift apart, this case stops demonstrating what it claims.
    """
    razorpay.write_batch_csvs(entities, tmp_path)
    (tmp_path / "bank_statement.csv").write_text(
        SAMPLE_BANK.read_text(encoding="utf-8"), encoding="utf-8")

    settlements = {s.transaction_id: s for s in razorpay.settlements_from_recon(entities)}
    banks = {b.bank_row_id: b for b in load_batch(tmp_path).banks}
    refund = next(e for e in entities if e["type"] == "refund")

    shortfall = settlements["setl_RZB"].net_paise - banks["BNK_R0002"].credit_paise
    assert shortfall == refund["amount"]


# --------------------------------------------------------------------------
# The network boundary
# --------------------------------------------------------------------------

def test_missing_credentials_raise_rather_than_falling_back(monkeypatch):
    """A silent fallback to the fixture would let a live demo show canned data.

    DEVLOG entry 6 is the same class of bug: a degradation path that produced
    something indistinguishable from the real thing.
    """
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    with pytest.raises(razorpay.RazorpayError, match="not set"):
        razorpay.fetch_recon_report(2026, 7)


def test_a_live_key_is_refused_by_default(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_ABCDEFGH")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    with pytest.raises(razorpay.RazorpayError, match="live account"):
        razorpay.fetch_recon_report(2026, 7)


def test_test_mode_keys_are_recognised():
    assert razorpay.is_test_mode("rzp_test_ABC123")
    assert not razorpay.is_test_mode("rzp_live_ABC123")


def test_both_envelope_shapes_are_accepted():
    items = [{"entity_id": "x"}]
    assert razorpay.normalise_payload({"count": 1, "items": items}) == items
    assert razorpay.normalise_payload(items) == items
    with pytest.raises(razorpay.RazorpayError):
        razorpay.normalise_payload("not a collection")


def test_the_envelope_the_live_api_actually_returns():
    """The exact body observed from settlements/recon/combined, HTTP 200.

    Captured from a live test-mode account with no settlement history. Pinning
    the real bytes matters because `normalise_payload` returns [] both for a
    valid empty collection and for a shape it does not recognise - so "we called
    it and got no rows" would otherwise be indistinguishable from "we called it
    and silently failed to parse the answer". This asserts the `items` key is
    genuinely there.
    """
    observed = json.loads('{"entity":"collection","count":0,"items":[]}')
    assert "items" in observed
    assert razorpay.normalise_payload(observed) == []


def test_core_cannot_import_the_network_adapter():
    """Belt and braces with test_layer_separation - stated here too because this
    is the module that would break the offline guarantee if it ever leaked in."""
    for path in (ROOT / "core").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("integrations"), f"{path.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("integrations"), f"{path.name}"


# --------------------------------------------------------------------------
# Credential loading
# --------------------------------------------------------------------------

def test_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_ABC")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "shhh")
    assert razorpay.credentials() == ("rzp_test_ABC", "shhh")

    # A half-configured pair is not credentials. Returning the key alone would
    # produce an auth failure that reads like a bad key rather than a missing one.
    monkeypatch.delenv("RAZORPAY_KEY_SECRET")
    assert razorpay.credentials() is None


def test_the_suite_never_reads_a_developers_dotenv(tmp_path, monkeypatch):
    """Hermeticity: a real key on disk must not leak into the tests.

    `envfile.load_env_files` short-circuits under pytest precisely so that a
    populated .env.local cannot turn a test run into a live API call against
    someone's real Razorpay account. This asserts that guard rather than trusting
    it, because the failure mode is silent and expensive.
    """
    import envfile

    monkeypatch.setattr(envfile, "REPO_ROOT", tmp_path)
    (tmp_path / ".env.local").write_text(
        "RAZORPAY_KEY_ID=rzp_test_LEAKED\nRAZORPAY_KEY_SECRET=leaked\n",
        encoding="utf-8")
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)

    envfile.load_env_files(force=True)

    assert os.environ.get("RAZORPAY_KEY_ID") is None, (
        "a .env.local on disk leaked into the test environment")
    assert razorpay.credentials() is None
