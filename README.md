# ReconAgent

**Three-way payment reconciliation with an honest exception list.**

Razorpay AI Buildathon — Track 04, AI Finance Controller.

Every merchant on a payment gateway has to reconcile three systems of record each
settlement cycle: the gateway's settlement report, the bank statement, and their
own order ledger. They never agree cleanly. Fees and GST come off the top,
settlements land T+1 or T+2, one payout arrives as three credits, refunds land
partially, webhook retries duplicate rows, and paisa-level rounding puts amounts
just off. Today a person does this in a spreadsheet.

ReconAgent reconciles the three sources, reports measured accuracy against ground
truth it never sees, and for everything it cannot resolve produces a
human-readable investigation of **why** — including the cases where the right
answer is *"this cannot be determined, send it to a person."*

---

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
```

That is the whole setup. **No API key is required** and no LLM SDK is installed —
the deterministic engine, every metric, the exception list, the cash position and
the Q&A all run offline, replaying 104 committed reasoning traces.

```bash
streamlit run app.py                       # the UI
python run_demo.py --dataset holdout       # the sealed evaluation set
python run_demo.py --ask "why didn't pay_f7atwyam1n reconcile?"
python -m pytest tests/ -q                 # 191 tests
python -m evaluation.sensitivity           # threshold trade-off sweep
python benchmark.py                        # throughput and accuracy vs batch size
```

To regenerate reasoning against a live model, install a provider and set a key:

```bash
pip install -r requirements-llm.txt
cp .env.example .env.local     # add OPENAI_API_KEY or ANTHROPIC_API_KEY
python run_demo.py
```

---

## Results

Measured against ground truth the engine and the model never see. `dev` is the
tuning set. `holdout` was generated from a different seed with a harder case mix,
a wider settlement lag and a bank narration format the parser had never seen, and
was **opened once**, after thresholds were frozen — provable from git history:
`core/config.py` has not changed since the first commit.

| | dev | holdout |
|---|---|---|
| rows / groups | 252 / 85 | 290 / 97 |
| auto-match rate | 63.5% | 49.5% |
| **precision** | **100%** | **100%** |
| **recall** | **100%** | **84.2%** |
| F1 | 100% | 91.4% |
| **false positives** | **0** | **0** |
| false negatives | 0 | 9 |
| adversarial pairs conflated | **0 / 5** | **0 / 9** |
| exceptions correctly held | 31 / 31 | 40 / 40 |
| exceptions correctly categorised | 100% | 100% |
| throughput (deterministic) | see [At scale](#at-scale) | see [At scale](#at-scale) |

> **These two sets are too small to measure the false-positive rate, and the
> throughput figure is meaningless.** Both were caught by scaling the benchmark to
> 25,000 records — see [At scale](#at-scale) below, which supersedes the last two
> rows of this table. The numbers above are accurate for these batches; they are
> just not evidence of what I originally claimed they were.

**The held-out recall gap is real and is not being hidden.** All 9 misses are
split settlements whose legs land at T+3 or T+4, in a batch that models a bank on
a slower cycle than the engine's configured T+2 window. The engine refused rather
than mis-binding, so precision held at 100%. The window was **not** widened to
close the gap — that would be tuning on the evaluation set. `evaluation/sensitivity.py`
measures what that knob costs: at T+4 held-out recall reaches 96% with no
precision loss on this data. The right value is a property of the bank you
reconcile against, not of a test set.

### The two errors are not the same error

Precision and recall are reported separately, and so are their causes, because
collapsing them into one accuracy number lets a system trade the expensive
mistake for the cheap one and look better for it.

- A **false positive** binds money to the wrong counterpart. It produces books
  that look clean and are wrong, and nobody goes looking. Zero on both sets above —
  but see [At scale](#at-scale): the true rate is around 0.5% of settlements, and
  these sets are too small to detect it.
- A **false negative** hands a matchable row to a human. It costs a few minutes.
  A system that refuses when uncertain will deliberately incur more of them.

### At scale

The figures above come from 252 and 290 records. That is enough to demonstrate the
case types and nowhere near enough to measure an error rate, so I scaled the same
seeded generator and re-ran everything. `benchmark.py` reproduces this.

**Finding 1 — the false-positive rate was about 0.5%, and my headline sets could
not see it.** Two independent seeds, before the fix described below:

| rows | settlements | false positives | FP rate | precision |
|---|---|---|---|---|
| 255 | 87 | 0 / 0 | 0.00% | 100% |
| 1,008 | 346 | 0 / 0 | 0.00% | 100% |
| 5,042 | 1,727 | 10 / 6 | 0.58% / 0.35% | ~99.1% |
| 24,967 | 8,547 | 46 / 35 | 0.54% / 0.41% | ~99.1% |

The rate is roughly constant. It reads as exactly zero on the dev set for an
uninteresting statistical reason: 0.5% of 87 settlements is 0.44 expected errors,
so **observing zero is the most likely outcome even when the true rate is 0.5%**.
"100% precision, 0 false positives" was never evidence of perfection — it was a
sample too small to detect the error rate that was there all along.

**Where they come from.** Every one is tier 3 (`amount_date_window`); none are
splits and none are the ambiguity guard. A settlement that should be a split, or
should have no counterpart at all, gets bound to a *single* unrelated credit that
coincidentally lands within ±50 paise and the 3-day window. The ambiguity guard
fires on two or more candidates and had no defence against exactly one.

**The fix: a coincidence guard.** An amount-and-date match carries no identifier,
so it is worth only as much as the odds against a coincidence. The engine now
scores each one:

```
expected coincidences = neighbours within ±Rs 50  x  (delta / Rs 50)  x  ((lag + 1) / window)
```

An exact match on the settlement date with nothing else nearby scores ~0 and is
accepted. One sitting mid-tolerance, two days late, in a crowded field scores
high - which is exactly what a coincidence looks like, because coincidental deltas
spread evenly across the tolerance band while real ones cluster at zero. Above
0.05 the match is held for review, and the credit is left available to a
settlement that can account for it better.

| 25,000 rows | false positives | precision | recall |
|---|---|---|---|
| seed 7, guard off | 46 | 98.98% | 83.54% |
| **seed 7, guard on** | **24** | **99.46%** | 83.26% |
| seed 99, guard off | 35 | 99.22% | 83.63% |
| **seed 99, guard on** | **22** | **99.51%** | 83.33% |

Seed 99 was never used to design it. The guard removes roughly 40% of false
positives for 0.3 percentage points of recall - a trade worth taking, since
binding money to the wrong counterpart costs far more than asking a human to look.
It is **dormant at small batch sizes**: dev and holdout are byte-identical with it
on and off, which a test asserts.

The residual ~0.3% is not separable with the features available. Amount and date
alone cannot distinguish a real payout from a coincidence once a batch is dense
enough; that needs narration text or historical pairing, which this synthetic data
does not carry.

**Finding 2 — the adversarial claim survives scale.** 0 conflations out of **495**
near-duplicate pairs at 25,000 rows. That claim *is* well-powered, and it is the
one the refusal design was built to support.

**Finding 3 — matching was quadratic, so the throughput figure was fiction.**
The 141,000 rows/s in the results table was measured over 1.8 ms — interpreter
noise, not an algorithm. Measured properly it was **2,721 rows/sec at 25,000
records**, and falling. Tiers 2 through 5 each scanned every unclaimed credit for
every settlement: 21.5 million amount comparisons on a single 25,000-row batch.

Three changes, none of which alters a single result:

- **`core/index.py`** — credits bucketed by value date, each bucket sorted by
  amount, so a candidate lookup is two binary searches instead of a full scan.
- **Tier 2** — prefix-matching truncated UTRs by binary search over a sorted list
  rather than scanning every pending settlement UTR (1,980 × 17,000 comparisons
  on a 50,000-row batch).
- **Subset-sum** — the split search walked past every leg too large to fit; it now
  skips that prefix with a bisect. This was 71% of total time at 100,000 rows.

| rows | before | after | speedup | rows/sec after |
|---|---|---|---|---|
| 5,042 | 0.75 s | 0.40 s | 1.9x | 12,485 |
| 24,967 | 9.18 s | **0.85 s** | **10.8x** | **29,481** |
| 49,853 | ~37 s (est.) | **2.09 s** | ~18x | 23,864 |
| 100,058 | ~147 s (est.) | **6.48 s** | ~23x | 15,447 |

Scaling is near-linear to 50,000 rows and mildly superlinear beyond. The residual
cost is subset-sum itself: as a batch densifies, more credits fall inside each
settlement's window and the split search has more ground to cover. That is
inherent to detecting splits, not an implementation accident.

**Every optimisation was verified byte-identical** — same statuses, confidences,
linked IDs, rule traces and near misses across dev, holdout and two generated
batches, 12,940 groups in total. One early version differed on a *single line of
one rule trace* out of 12,940 groups; it was reverted, because a faster engine
that quietly edits its own audit trail is not a faster engine.

```bash
python benchmark.py --rows 250 5000 25000
```

---

## Architecture

```mermaid
flowchart TB
    S["settlement_report.csv<br/><i>gateway payouts</i>"]
    B["bank_statement.csv<br/><i>semi-structured narration</i>"]
    L["internal_ledger.csv<br/><i>merchant orders</i>"]

    S & B & L --> LOAD["<b>core/loader</b><br/>rupees to integer paise<br/>UTR recovery + provenance"]

    LOAD --> ENGINE

    subgraph ENGINE["core/matcher — deterministic, no LLM, no network"]
        direction TB
        T0["0 · duplicates<br/><i>webhook retries, before anything matches</i>"]
        T1["1 · exact UTR<br/><i>+ re-verify the amount</i>"]
        T2["2 · repaired UTR<br/><i>unique prefix only</i>"]
        T3["3 · amount + date<br/><i>constraint propagation</i>"]
        T4["4 · split settlement<br/><i>bounded subset-sum</i>"]
        T5["5 · unresolved<br/><i>+ near misses as evidence</i>"]
        T0 --> T1 --> T2 --> T3 --> T4 --> T5
    end

    ENGINE -->|matched / split / duplicate| TRAIL
    ENGINE -->|exceptions only| ROUTE{"<b>agent/reasoner</b><br/>how hard is it?"}

    ROUTE -->|"engine had a candidate<br/>and declined"| DEEP["<b>agent/investigator</b><br/>bounded multi-turn loop<br/>chooses its own queries"]
    ROUTE -->|"trivial absence"| SHALLOW["one-shot explanation<br/>batched, 6 per call"]

    DEEP <-->|"read-only questions"| TOOLS["<b>agent/tools</b><br/>search credits · get row<br/>find UTR · batch summary<br/><i>no tool can change a match</i>"]

    DEEP --> TRAIL["<b>audit/trail</b><br/>append-only JSONL<br/>decisions + every agent query"]
    SHALLOW --> TRAIL

    TRAIL --> QA["<b>agent/qa</b><br/>retrieval over the trail"]
    ENGINE --> EVAL["<b>evaluation/metrics</b><br/>vs ground truth"]
    ENGINE --> CASH["<b>cash/position</b><br/>forward cash view"]

    GT[("ground_truth.json")] -.->|"read ONLY here"| EVAL

    style ENGINE fill:#e8f4ea,stroke:#2d6a3e
    style DEEP fill:#eef2fb,stroke:#3d5a99
    style SHALLOW fill:#f4f4f7,stroke:#777
    style TOOLS fill:#fdf6e8,stroke:#a07a2c
    style GT fill:#fbeeee,stroke:#994444
```

Cascade order is a correctness mechanism, not an optimisation. Resolving strong
identifiers **before** reaching for amounts is exactly what defeats near-duplicate
transactions: tier 1 consumes the identified leg, so by the time amount matching
runs the ambiguity is gone. Swap those two tiers and false positives appear.

---

## Where I deliberately did not use an LLM

**Matching.** All of it. The engine is pure Python over integer paise: no model
call, no network, no randomness. Same CSVs plus same config produce byte-identical
output. A language model cannot move a match, lower a confidence, or add a link.

This is not a convention. It is enforced:

- `tests/test_layer_separation.py` parses every `core/*.py` AST and fails the
  build if it imports any model SDK, HTTP client, or the `agent` package.
- It runs a full reconciliation with `socket.socket` replaced by something that
  raises, proving nothing phones home.
- It runs another with `anthropic` and `openai` made unimportable.
- `tests/test_reasoning_layer.py` snapshots every field of every result, runs the
  reasoning layer, and asserts nothing moved.
- `tests/test_llm_provider.py` asserts reconciliation output is identical under
  Anthropic, OpenAI, and no provider at all.

**The result:** 58 of 85 groups (65%) never reach a model at all.

### Where I did use one

Investigating exceptions, and phrasing Q&A answers — and the effort spent scales
with difficulty, the same way the matching engine's does.

| | cases (dev) | what happens |
|---|---|---|
| never reaches a model | 58 | matched confidently by rules |
| one-shot explanation | 8 | a sentence is enough — an orphan credit resembling nothing |
| **multi-turn investigation** | **23** | the engine had a candidate and declined; the agent gets tools |

**The investigation loop is what makes this an agent rather than a prompt.** For
the hard cases the model gets **read-only tools** over the batch — search credits
by amount and date, pull a bank row, look up a settlement, search UTR fragments —
and up to four turns. It chooses each query, sees real results, and decides when
it has enough.

Every tool is a *question*. There is deliberately no tool that creates a link,
changes a status or resolves anything, so no sequence of agent actions can alter a
reconciliation outcome (`test_no_tool_can_mutate_the_reconciliation`). The loop is
hard-bounded, and failing to converge escalates rather than guessing
(`test_the_loop_is_bounded`).

Every query and every result is written to the audit trail, so a reviewer can walk
the same path the agent walked.

**Grounding is measured:** across 104 traces the model wrote 264 record IDs, and
**zero** were absent from the evidence it was shown.

The output schema requires a `sufficient_evidence` boolean, and the prompt states
plainly that "there is not enough here to decide" is a correct and valued answer.
Without that, a model asked to explain an unmatched transaction reliably invents a
match — it reads the task as *find the answer* rather than *judge whether an
answer exists*.

---

## The case worth looking at

`pay_f7atwyam1n` — ₹48,615.93 settled 2026-07-17, no UTR on any bank row.

```
tier3 amount+date: 2 credits satisfy both thresholds (BNK_00032, BNK_00056).
Refusing to guess - picking the closest would be a coin flip reported as a match.
```

Both candidates match the amount **exactly**. Both landed on the **right date**.
The engine claims neither, and the agent agrees:

> The Rs 48,615.93 net settlement dated 2026-07-17 has two identical same-day
> candidate credits, BNK_00032 and BNK_00056, and neither carries UTR
> 327723997757. Their different narrations do not establish which credit belongs
> to this settlement.
>
> → `escalate_to_human` · confidence **high** · `sufficient_evidence: false`

High confidence in the judgement that it *cannot* be decided. A matcher that
picked the closer amount here would be right half the time and report certainty
every time. There are 5 such pairs in dev and 9 in holdout; **zero** were
conflated.

It did not simply accept the engine's finding — it went and looked:

```
turn 1  credits_near_settlement(pay_f7atwyam1n)
        "inspect all credits around this settlement to determine whether either
         same-day identical amount has distinguishing evidence"
turn 2  get_credit(BNK_00032)
        "inspect the full bank-row details to see whether any reference, posting
         metadata, or linkage distinguishes it from the other identical candidate"
turn 3  conclude
        "both bank rows are identical on the available matching facts"
```

The planted near-miss case is the same story with a different ending. There, the
agent searched wider than the engine's window, found the single candidate that
exists, checked its narration for an explanation, searched the settlement's UTR
across the whole batch — and still declined:

> BNK_00027 is the only plausible nearby unmatched credit... It is Rs 1.06 short
> and six days late, and neither its narration nor the UTR search explains the
> difference. Do not match automatically.

**An agent that refuses what it was never allowed to look for is restating its
input. An agent that searches, finds the one candidate, and still declines is
exercising judgement** — and because every query is in the audit trail, that
judgement is checkable rather than asserted.

The synthetic generator builds these deliberately, with *identical* net amounts —
so no threshold setting anywhere can separate them
(`test_ambiguous_twins_are_undecidable_at_any_tolerance`).

### Refusal is not a dead end

The agent recommended `auto_resolve` on 8 exceptions, all partial refunds:

> BNK_00042 carries the same UTR as the settlement and was credited on
> 2026-07-08, but its Rs 4,009.22 credit is Rs 1,858.69 below the Rs 5,867.91
> expected net. The linked ledger order is marked partially refunded, so the
> shortfall is consistent with a legitimate partial refund.

The engine still holds all 8 as exceptions. The agent explained away 8 benign
items **without moving a single match** — which is the entire design in one line.

---

## Cash position

Completes the track's second half — *"run the books **and the cash position**."*
Every rupee is attributed to one bucket, and the three in-account buckets are
asserted to sum exactly to the bank total.

| In the account | | Expected, not received | |
|---|---|---|---|
| reconciled & explained | ₹753,662.83 | in flight (within T+2) | ₹104,849.98 |
| arrived, unreconciled | ₹106,941.22 | overdue | ₹215,847.84 |
| unattributed inflows | ₹155,490.18 | | |
| | **74% explained** | **needs a human today** | **₹478,279.24** |

"In flight" and "overdue" are kept apart deliberately: a settlement at T+1 with no
credit is normal; the same settlement at T+9 is an incident. The engine cannot
tell those apart — both are simply an absent credit — so that temporal judgement
is made by the cash layer, which knows the settlement window.

---

## Test data

Synthetic, seeded, reproducible: `python -m data.generator`.

Every case is constructed to defeat one specific shortcut.

| case | dev | what it defeats |
|---|---|---|
| clean | 10 | baseline |
| fee deduction | 15 | comparing ledger gross to bank credit |
| timing lag | 10 | same-day joins, exact-string UTR joins |
| split settlement | 8 | one-to-one matching |
| partial refund | 8 | trusting an ID match without re-checking value |
| duplicate | 5 | double-counting webhook retries |
| rounding | 5 | exact-amount equality |
| adversarial (resolvable) | 6 | amount-first matching — cascade order saves it |
| **adversarial (ambiguous)** | **4** | **anything. Identical amounts, no UTR — must be refused** |
| pending settlement | 5 | reading "not yet arrived" as "missing" |
| unmatchable | 9 | force-matching when no counterpart exists |

30 integrity tests assert the data is what ground truth claims: fee arithmetic
closes in integer paise, split legs sum to the net, adversarial twins really are
indistinguishable, and the planted near-miss is genuinely unmatchable against
every settlement in the batch. Bank row IDs are assigned **after** a global
shuffle, so split legs are not contiguous — otherwise adjacency would leak the
grouping and split detection would be measuring the generator.

---

## Honest limitations

**The data is synthetic and I wrote it.** I designed the messiness and then built
a matcher for it, which bounds what 100% on dev means. Mitigations: a held-out set
from a different seed with a different mix, thresholds frozen in git before it was
opened, and a sensitivity sweep across the parameter space. It is a real
mitigation, not a cure. Real bank statements are messier than anything here.

**The held-out recall gap is unfixed by choice.** 84.2%, all split settlements
beyond the configured window. See above.

**The ledger records refund status but not refund amount.** The agent found this
itself: on a partial refund it can see the order is marked refunded, but nothing
lets it confirm the shortfall reconciles, so it escalates. A real system would
join a refunds table. That the agent noticed and said so is the behaviour I want;
the missing column is a limitation of my synthetic sources.

**Ambiguous cases are refused, not resolved.** With more signal — narration
tokens, customer names in the bank reference, historical pairing — several would
be decidable. The engine currently uses amount, date and UTR only.

**The reasoning layer is advisory and unverified.** Its recommendations are not
checked against ground truth, because it does not make decisions. Grounding is
measured (0/264 ungrounded IDs); usefulness is not.

**Scale is now tested, and it found two things.** Matching is quadratic, so the
throughput headline was a ~50x overstatement; and the false-positive rate is ~0.5%
of settlements rather than zero, which my 87-settlement dev set had no power to
detect. Both are measured and addressed in [At scale](#at-scale). The false-positive rate
is roughly halved by a coincidence guard validated on an unseen seed; the residual
~0.3% is not separable from amount and date alone. Matching is ~10x faster and
near-linear to 50,000 rows, verified byte-identical. Beyond that it is still
mildly superlinear, and the remaining cost is subset-sum, which is inherent to
split detection rather than an implementation accident.

---

## What I would build next

1. **Bound the split search on dense batches.** Subset-sum is the remaining
   superlinear cost at 100,000 rows. Capping the candidate pool would fix it, and
   would also cut the coincidental splits that grow with density - but it changes
   results, so it needs its own before-and-after rather than being slipped in.
2. **More signal for lone tier-3 matches.** The coincidence guard halves them;
   the rest need evidence this data does not carry. Narration tokens and customer
   names in the bank reference would separate a real payout from a coincidence
   where amount and date cannot.
3. **Per-bank configuration profiles.** The held-out gap is a configuration
   problem, and the settlement window should be learned per counterparty rather
   than set globally.
4. **Feed resolved exceptions back as signal.** Once a human pairs an orphan
   credit with an overdue settlement, that pairing is training data for narration
   matching — the highest-value unused signal in the data.
5. **Currency and multi-entity**, where rounding and FX make the tolerance
   question genuinely hard rather than a fixed 50 paise.
6. **Verify the reasoning layer.** Have a second model, or a human queue, check a
   sample of investigations so its usefulness is measured rather than assumed.

---

## Layout

```
core/         deterministic engine — no LLM, no network, integer paise
  index.py      date-bucketed, amount-sorted credit lookup
  config.py     thresholds, frozen in git since the first commit
  loader.py     CSV to typed records, UTR recovery with provenance
  matcher.py    the tiered cascade
  splits.py     bounded subset-sum, refuses when the answer is not unique
agent/        the reasoning layer — explains, never matches
  llm.py        provider shim: Anthropic or OpenAI, chosen from the environment
  prompts.py    system prompt + output schema
  tools.py      read-only query surface the agent investigates with
  investigator.py  bounded multi-turn loop for the hard cases
  reasoner.py   routes each exception to the deep or the cheap path
  qa.py         retrieval over the audit trail
  cache/        104 committed traces, so the repo demos offline
audit/        append-only JSONL decision trail
evaluation/   metrics + threshold sensitivity — the only reader of ground truth
benchmark.py  throughput and accuracy as a function of batch size
cash/         forward cash position
data/         seeded generator, dev and holdout batches
tests/        191 tests
DEVLOG.md     what actually broke, and what I did about it
```

`DEVLOG.md` is worth reading alongside the code. It records six real failures from
this build, including a cache-poisoning bug that would have shipped 31 HTTP 401s
into this repository as the project's demonstration of agent reasoning.
