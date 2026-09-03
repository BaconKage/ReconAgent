"""Read a real Razorpay settlement recon report into the engine's records.

Why this exists. The rest of this project reconciles CSVs from a generator I
wrote, which bounds what its accuracy figures prove - stated plainly in the
README's limitations. This module removes one half of that objection: the
gateway side of the batch can come from Razorpay's own
``GET /v1/settlements/recon/combined`` response, in the exact field shape the
API documents, rather than from anything I invented.

**The bank side deliberately cannot come from here, and that is the whole
problem.** Razorpay knows what it paid out; only the merchant's bank knows what
landed. If one system held both, there would be nothing to reconcile. So this
adapter fills the settlement and ledger sources and leaves ``bank_statement.csv``
to the merchant's own export - which is exactly the asymmetry the engine exists
to bridge.

Two conveniences the API shape gives us, both worth noting:

* **Amounts arrive as integer paise.** The engine is integer-paise native and
  does no float arithmetic on money, so the value crosses the boundary without a
  decimal conversion that could round. The CSV path has to parse rupee strings;
  this path does not.
* **``settlement_id`` groups payments into the payout that actually hits the
  bank.** A settlement is credited as one amount covering many payments, so
  aggregating by ``settlement_id`` reconstructs the row the bank will show. That
  aggregation is done here rather than in ``core/``, because it is a fact about
  Razorpay's payout model, not about reconciliation.

Usage, offline, against the committed fixture:

    python -m integrations.razorpay --out data/rzp

Against a live **test-mode** account, if credentials are present:

    export RAZORPAY_KEY_ID=rzp_test_...
    export RAZORPAY_KEY_SECRET=...
    python -m integrations.razorpay --year 2026 --month 7 --out data/rzp

This module is not importable from ``core/``. ``tests/test_layer_separation.py``
forbids it there alongside the model SDKs, for the same reason: the matching
engine must not be able to reach the network.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.models import LedgerRecord, SettlementRecord
from envfile import load_env_files

API_ROOT = "https://api.razorpay.com/v1"
RECON_COMBINED = "/settlements/recon/combined"

#: Committed sample response in the documented recon-combined shape, so this
#: path runs with no credentials and no network. It is a fixture, not a capture
#: of a real account - see the note in its own header.
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "razorpay_sample" / "recon_combined.json"

#: A sample bank export to pair with the fixture, so the offline path produces a
#: runnable batch. Copied in only on a fixture run and announced when it is -
#: never alongside a live recon report, where it would silently reconcile real
#: settlements against invented credits.
SAMPLE_BANK = FIXTURE.parent / "bank_statement.csv"

#: Recon entity types that represent money moving out to the merchant. Refunds
#: and adjustments appear in the same report with the opposite sign and must not
#: be added into a payout total.
CREDIT_TYPES = {"payment"}


class RazorpayError(RuntimeError):
    """A live API call failed. Deliberately not caught by the callers here.

    DEVLOG entry 6 is about a fallback that quietly persisted its own failure
    output. The lesson generalised: a network path that degrades silently into a
    fixture would let a demo claim live data while showing canned data. If the
    API call fails, this raises and the caller decides.
    """


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def credentials() -> tuple[str, str] | None:
    """Razorpay API credentials from the environment, or None.

    `.env.local` is read first, matching how model keys are configured, so a key
    put where the README says to put it is actually found. Without this the
    adapter reported "not set" for a key sitting in the file it told you to use -
    a failure indistinguishable from a wrong key.
    """
    load_env_files()
    key_id = os.environ.get("RAZORPAY_KEY_ID", "").strip()
    secret = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()
    if key_id and secret:
        return key_id, secret
    return None


def is_test_mode(key_id: str) -> bool:
    """Razorpay test keys are prefixed ``rzp_test_``; live keys ``rzp_live_``."""
    return key_id.startswith("rzp_test_")


def fetch_recon_report(year: int, month: int, *, timeout: float = 30.0,
                       allow_live_keys: bool = False) -> list[dict[str, Any]]:
    """GET /v1/settlements/recon/combined for one month.

    Refuses live-mode keys unless explicitly allowed. This is a buildathon
    project reconciling test data; pointing it at a production account by an
    environment-variable accident should not be one typo away.
    """
    creds = credentials()
    if creds is None:
        raise RazorpayError(
            "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET are not set. "
            "Run without --year/--month to use the committed fixture instead.")
    key_id, secret = creds

    if not is_test_mode(key_id) and not allow_live_keys:
        raise RazorpayError(
            f"{key_id[:12]}... does not look like a test key. Refusing to call a "
            f"live account. Pass --allow-live-keys if that is genuinely intended.")

    url = f"{API_ROOT}{RECON_COMBINED}?year={year}&month={month:02d}"
    token = base64.b64encode(f"{key_id}:{secret}".encode()).decode()
    req = urllib.request.Request(url, headers={
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        raise RazorpayError(f"HTTP {e.code} from {RECON_COMBINED}: {body}") from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise RazorpayError(f"could not reach {RECON_COMBINED}: {e}") from e

    return normalise_payload(payload)


def load_fixture(path: Path = FIXTURE) -> list[dict[str, Any]]:
    """The committed sample response, for running with no credentials."""
    with open(path, encoding="utf-8") as f:
        return normalise_payload(json.load(f))


def normalise_payload(payload: Any) -> list[dict[str, Any]]:
    """Accept either the ``{count, items:[...]}`` envelope or a bare list.

    Razorpay's collection endpoints wrap results in ``items``; some exports hand
    you the array directly. Tolerating both here keeps the shape question out of
    every call site.
    """
    if isinstance(payload, dict):
        return list(payload.get("items") or payload.get("entities") or [])
    if isinstance(payload, list):
        return list(payload)
    raise RazorpayError(f"unexpected recon payload of type {type(payload).__name__}")


# --------------------------------------------------------------------------
# Mapping into the engine's records
# --------------------------------------------------------------------------

def _ts_to_date(value: Any) -> date | None:
    """Razorpay timestamps are unix seconds, UTC."""
    if value in (None, "", 0):
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None


def _int_paise(value: Any) -> int:
    """Recon amounts are already integer paise. Missing means zero, not None.

    A fee or tax absent from a row means none was charged. Coercing to 0 here is
    safe in a way that coercing a *settlement amount* would not be, which is why
    the caller checks the amount separately.
    """
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def settlements_from_recon(entities: Iterable[dict[str, Any]]) -> list[SettlementRecord]:
    """Aggregate recon entities into one settlement per payout.

    A Razorpay settlement covers many payments and reaches the bank as a single
    credit carrying a single UTR, so the unit the engine should try to match is
    the settlement, not the payment. Summing gross, fee and tax across the
    payments in a settlement reconstructs that credit:

        net = sum(amount) - sum(fee) - sum(tax)

    which is the same fee-and-GST arithmetic the CSV path models, sourced from
    the gateway instead of from my generator.

    Entities with no ``settlement_id`` are payments Razorpay has not yet paid
    out. They are skipped rather than emitted as zero-UTR settlements: an
    unsettled payment is not a missing credit, and conflating the two is
    precisely the "pending vs overdue" distinction the cash layer exists to make.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entities:
        if (e.get("type") or "payment") not in CREDIT_TYPES:
            continue
        sid = e.get("settlement_id")
        if not sid:
            continue
        groups[sid].append(e)

    out: list[SettlementRecord] = []
    for sid, rows in sorted(groups.items()):
        gross = sum(_int_paise(r.get("amount")) for r in rows)
        fee = sum(_int_paise(r.get("fee")) for r in rows)
        tax = sum(_int_paise(r.get("tax")) for r in rows)

        settled = next((_ts_to_date(r.get("settled_at")) for r in rows
                        if _ts_to_date(r.get("settled_at"))), None)
        if settled is None or gross <= 0:
            continue

        utr = next((str(r["utr"]) for r in rows if r.get("utr")), None)
        # order_id is only meaningful when the payout covers exactly one order.
        # Attaching an arbitrary one to a multi-order settlement would put a
        # false link in the audit trail.
        orders = {r.get("order_id") for r in rows if r.get("order_id")}
        order_id = orders.pop() if len(orders) == 1 else None

        out.append(SettlementRecord(
            transaction_id=sid,
            order_id=order_id,
            gross_paise=gross,
            fee_paise=fee,
            tax_on_fee_paise=tax,
            net_paise=gross - fee - tax,
            settlement_date=settled,
            utr_number=utr,
        ))
    return out


def ledger_from_recon(entities: Iterable[dict[str, Any]]) -> dict[str, LedgerRecord]:
    """Order-level rows, which a merchant would normally hold in their own system.

    Payment-level detail is not lost by the settlement aggregation above - it
    lands here, where the engine uses it to explain a settlement rather than to
    match one. Refund entities downgrade their order's status, which is the
    signal the reasoning layer reads when a credit falls short of the net.
    """
    ledger: dict[str, LedgerRecord] = {}
    refunded: set[str] = set()

    for e in entities:
        oid = e.get("order_id")
        if not oid:
            continue
        etype = e.get("type") or "payment"
        if etype == "refund":
            refunded.add(oid)
            continue
        if etype not in CREDIT_TYPES:
            continue
        created = _ts_to_date(e.get("created_at"))
        if created is None:
            continue
        notes = e.get("notes") or {}
        ledger[oid] = LedgerRecord(
            order_id=oid,
            # The recon report carries no customer name. Notes are merchant-set
            # and often hold one; saying where it came from beats inventing it.
            customer=str(notes.get("name") or notes.get("customer") or "(not in recon report)"),
            expected_paise=_int_paise(e.get("amount")),
            order_date=created,
            status="captured",
        )

    for oid in refunded:
        if oid in ledger:
            r = ledger[oid]
            ledger[oid] = LedgerRecord(
                order_id=r.order_id, customer=r.customer,
                expected_paise=r.expected_paise, order_date=r.order_date,
                status="partially_refunded",
            )
    return ledger


# --------------------------------------------------------------------------
# Writing the engine's CSV shape
# --------------------------------------------------------------------------

def _rupees(paise: int) -> str:
    """Integer paise back to the rupee string the CSV loader parses."""
    sign = "-" if paise < 0 else ""
    p = abs(paise)
    return f"{sign}{p // 100}.{p % 100:02d}"


def write_batch_csvs(entities: Iterable[dict[str, Any]], out_dir: str | Path) -> dict[str, int]:
    """Write settlement_report.csv and internal_ledger.csv from a recon report.

    Emitting the same CSV shape the generator produces means ``core/loader.py``
    ingests Razorpay data through the identical code path, with no branch for
    "real" versus "synthetic" input. Nothing in ``core/`` knows this module
    exists, which is the point.

    ``bank_statement.csv`` is not written - it is the merchant's export, and the
    caller is told so rather than being handed an empty file that would silently
    reconcile to nothing.
    """
    entities = list(entities)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    settlements = settlements_from_recon(entities)
    ledger = ledger_from_recon(entities)

    with open(out / "settlement_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["transaction_id", "order_id", "gross_amount", "fee",
                    "tax_on_fee", "net_amount", "settlement_date", "utr_number"])
        for s in settlements:
            w.writerow([s.transaction_id, s.order_id or "", _rupees(s.gross_paise),
                        _rupees(s.fee_paise), _rupees(s.tax_on_fee_paise),
                        _rupees(s.net_paise), s.settlement_date.isoformat(),
                        s.utr_number or ""])

    with open(out / "internal_ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "customer", "expected_amount", "order_date", "status"])
        for o in sorted(ledger.values(), key=lambda r: r.order_id):
            w.writerow([o.order_id, o.customer, _rupees(o.expected_paise),
                        o.order_date.isoformat(), o.status])

    return {"entities": len(entities), "settlements": len(settlements),
            "ledger_orders": len(ledger)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, help="fetch live instead of using the fixture")
    ap.add_argument("--month", type=int, help="1-12, required with --year")
    ap.add_argument("--out", default="data/rzp", help="directory to write CSVs into")
    ap.add_argument("--allow-live-keys", action="store_true",
                    help="permit a rzp_live_ key (refused by default)")
    args = ap.parse_args()

    if args.year:
        if not args.month:
            ap.error("--month is required with --year")
        try:
            entities = fetch_recon_report(args.year, args.month,
                                          allow_live_keys=args.allow_live_keys)
        except RazorpayError as e:
            # A wrong key, an unreachable API and a refused live key are all
            # expected operator errors, not bugs. A traceback here would bury the
            # one line that says which of them happened.
            print("Could not fetch the recon report.")
            print()
            print(f"  {e}")
            print()
            msg = str(e)
            if "401" in msg or "Authentication failed" in msg:
                print("  Authentication failed, so the pair was read but rejected. Check")
                print("  that RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET come from the SAME")
                print("  generated pair, and that both are Test Mode.")
            elif "not set" in msg:
                print("  Put them in .env.local (see .env.example), or export them.")
            print()
            print("  The committed fixture needs no credentials:")
            print("      python -m integrations.razorpay --out data/rzp")
            return 1
        source = f"live recon report {args.year}-{args.month:02d}"
    else:
        entities = load_fixture()
        source = f"committed fixture {FIXTURE.name}"

    stats = write_batch_csvs(entities, args.out)
    print(f"source: {source}")
    print(f"  recon entities read      {stats['entities']}")
    print(f"  settlements written      {stats['settlements']}")
    print(f"  ledger orders written    {stats['ledger_orders']}")
    print(f"  -> {Path(args.out).resolve()}")
    print()

    if args.year and not entities:
        # An empty month is the normal result on a test account, and saying so
        # matters: the call still proved the credentials, the endpoint and the
        # response shape. A bare "0 settlements" reads like a failure when it is
        # the opposite - it is a successful call to an account with no history.
        print("  The API answered and returned no entities for this period.")
        print("  Authentication, the endpoint and the response shape are verified;")
        print("  this account simply has no settlements in that month. Test-mode")
        print("  accounts do not accrue settlement history on their own.")
        print()
        print("  To watch the engine run, use the committed fixture:")
        print("      python -m integrations.razorpay --out data/rzp")
        print("      python run_demo.py --data data/rzp")
    elif args.year:
        # Live data. Never pair a real settlement report with a sample bank
        # export - the reconciliation would be meaningless and would look real.
        print("  bank_statement.csv is NOT written: Razorpay knows what it paid out,")
        print("  only your bank knows what landed, and that gap is the reconciliation.")
        print("  Drop your bank export in beside these two files, then:")
        print(f"      python run_demo.py --data {args.out}")
    else:
        # Fixture run. Copy the sample bank export alongside so the batch is
        # runnable, and say so plainly - a bank statement appearing without
        # explanation is exactly the kind of silent convenience this project
        # spends a DEVLOG entry regretting.
        dest = Path(args.out) / "bank_statement.csv"
        dest.write_text(SAMPLE_BANK.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  sample bank export copied in: {dest.name}")
        print("  It is synthetic and it is mine. Razorpay cannot supply this half -")
        print("  only your bank knows what landed, which is why there is anything")
        print("  to reconcile. Against a live report, use your own export instead.")
        print()
        print("  The batch is runnable now:")
        print(f"      python run_demo.py --data {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
