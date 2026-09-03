# Architecture

Razorpay AI Buildathon — Track 04, AI Finance Controller.

This document is the design rationale: the boundaries, why they are where they
are, and what each one is defending against. The data-flow diagram, the measured
results and the reproduction commands live in [README.md](README.md); this file
does not restate any number, so the two cannot drift apart.

---

## The shape of the problem

Reconciliation looks like a join and is not one. Three systems describe the same
money and none of them agree:

| source | what it knows | what it does not |
|---|---|---|
| settlement report | what the gateway paid out, gross, fees, GST, UTR | whether it arrived |
| bank statement | what actually landed, and roughly when | what it was for |
| internal ledger | what the customer ordered | anything about settlement |

The gaps between them are not noise, they are structure: fees come off the top,
settlement lands T+1 or T+2, one payout can arrive as three credits, refunds land
partially, webhook retries duplicate rows, and rounding puts amounts just off.
A system that treats these as dirty data and forces a join produces books that
look clean and are wrong.

So the design question is not "how do I match more rows". It is **"how do I make
the system's confidence mean something"** — which turns out to be an argument
about where the decisions are allowed to be made.

---

## Three layers, and one boundary that matters

```
┌────────────────────────────────────────────────────────────────┐
│  core/          DETERMINISTIC                                  │
│  integer paise · no LLM · no network · no randomness           │
│  Same inputs + same config -> byte-identical output            │
│                                                                │
│  Decides: matched / split / duplicate / exception, and why     │
└────────────────────────────────────────────────────────────────┘
                    │  results are immutable past this line
                    ▼
┌────────────────────────────────────────────────────────────────┐
│  agent/         ADVISORY                                       │
│  reads exceptions · asks read-only questions · writes prose    │
│                                                                │
│  Decides: nothing                                              │
└────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────┐
│  audit/ evaluation/ cash/ integrations/   OBSERVERS            │
│  trail · scoring vs ground truth · forward cash · data sources │
└────────────────────────────────────────────────────────────────┘
```

**The boundary between the first two layers is the whole architecture.** A
language model in this system cannot move a match, change a status, lower a
confidence, or add a link. Not by convention — the reasoning layer is never given
a way to express such a change.

This is not caution for its own sake. Reconciliation is a domain where a
confident wrong answer is more expensive than no answer, because a false match
produces a clean-looking book that nobody audits. A model asked to explain an
unmatched transaction will, absent explicit instruction, reliably invent a match:
it reads the task as *find the answer* rather than *judge whether an answer
exists*. Putting it downstream of every decision means that tendency cannot cost
anything.

### How the boundary is enforced

Stating it in a README would be worth nothing — it decays the moment someone adds
a convenient import. So it is mechanical (`tests/test_layer_separation.py`):

- Every `core/*.py` module is parsed to an AST, and the build fails if it imports
  any model SDK, HTTP client, the `agent` package, or `integrations`.
- A full reconciliation runs with `socket.socket` replaced by something that
  raises — proving nothing phones home.
- Another runs with `anthropic` and `openai` made unimportable.
- `tests/test_reasoning_layer.py` snapshots every field of every result, runs the
  reasoning layer, and asserts nothing moved.
- `tests/test_llm_provider.py` asserts output is identical under Anthropic,
  OpenAI, and no provider at all.

The provider-swap test should be trivially true, since the engine never consults
the model. Asserting it anyway is the cheapest insurance on the project's central
claim: a vendor swap is the most direct way that claim could quietly become false.

---

## The matching cascade

`core/matcher.py` runs six tiers in a fixed order. **The order is a correctness
mechanism, not an optimisation.**

| tier | rule | defends against |
|---|---|---|
| 0 | duplicate detection, before anything matches | webhook retries double-counted |
| 1 | exact UTR, then re-verify the amount | trusting an identifier without checking value |
| 2 | repaired UTR, unique prefix only | truncated references in bank narration |
| 3 | amount + date, with constraint propagation | — |
| 4 | split settlement, bounded subset-sum | one-to-one assumptions |
| 5 | unresolved, with near misses recorded as evidence | silent failure |

Resolving strong identifiers **before** reaching for amounts is what defeats
near-duplicate transactions. Tier 1 consumes the identified leg, so by the time
amount matching runs the ambiguity is already gone. Swap tiers 1 and 3 and false
positives appear — the same data, the same thresholds, a worse answer.

Three properties worth naming:

**Constraint propagation, not greedy assignment.** Tier 3 binds a settlement only
when it has exactly one surviving candidate *and* no other settlement is also down
to that same candidate. Assignments shrink other candidate sets, so this iterates
to a fixed point. That is what lets one forced pairing cascade into resolving its
neighbour — and it is also what makes a lone leftover candidate diagnostic: if
propagation has run to fixed point and a settlement still has exactly one
unclaimed candidate, that can only mean another settlement wants it too. The
exception says so by name (`contested_candidate`) rather than reporting a generic
absence.

**Evidence without an identifier is worth only the odds against coincidence.** An
amount-and-date match carries no proof. Tier 3 therefore scores each one by
neighbourhood density, how loose the match is, and the lag, and holds it for
review above a threshold. Coincidental deltas spread evenly across the tolerance
band while real ones cluster at zero, which is what makes the score separable.

**Refusal is a first-class outcome.** Tier 5 does not mean "failed". An exception
carries a reason from a closed taxonomy, the rule trace that produced it, and the
near misses the engine considered and declined — so the output distinguishes
*"nothing resembled this"* from *"two settlements both wanted the same credit"*.
Those send an operator to different places, and an exception list that confuses
them is worse than no exception list, because it fails confidently.

---

## Data model decisions

**Money is integer paise, everywhere, with no float arithmetic in `core/`.** CSVs
carry rupee decimal strings because that is what real exports look like; the value
becomes an `int` the moment it crosses `core/loader.py` and stays one. Tolerances
are expressed in paise for the same reason. A reconciliation system that compares
money with floating point has a rounding bug it has not found yet.

**Provenance travels with recovered data.** A UTR dug out of narration text by
regex is not the same evidence as one sitting in a dedicated column, so
`BankRecord` carries how it was recovered. That reaches the audit trail, where a
reviewer can see whether a match rested on a clean field or a guess.

**The loader is tolerant and reports what it rejected.** Real bank exports contain
rows that do not parse, and a run that dies on one malformed line is useless. Bad
rows are collected, counted and surfaced — never silently dropped.

**Ground truth is readable by exactly one module.** `evaluation/` is the only code
permitted to open `ground_truth.json`, asserted by the layer-separation tests. The
engine cannot see the answers even accidentally.

---

## Where the model is used, and how its effort is scaled

Exceptions are routed by difficulty, the same way the matching engine scales its
own effort:

| route | when | what it gets |
|---|---|---|
| no model at all | the engine matched it confidently | — |
| one-shot explanation | trivial absence — a credit resembling nothing | one batched call |
| multi-turn investigation | the engine had a candidate and declined | read-only tools, bounded turns |

The investigation loop is what makes this an agent rather than a prompt. The model
chooses each query, sees real results from the actual batch, and decides when it
has enough.

**Every tool is a question.** There is deliberately no tool that creates a link,
changes a status, or resolves anything, so no sequence of agent actions can alter
a reconciliation outcome. The loop is hard-bounded, and failing to converge
escalates rather than guessing. Every query and every result is written to the
audit trail, so a reviewer can walk the same path the agent walked — which is what
makes its judgement checkable rather than asserted.

The output schema requires a `sufficient_evidence` boolean, and the prompt states
that *"there is not enough here to decide"* is a correct and valued answer. That
sentence is load-bearing: without it, the failure mode described above reasserts
itself immediately.

Grounding is measured rather than assumed — `verify_grounding.py` checks every
record ID the model wrote, including the arguments it chose for tool calls,
against the evidence it was actually shown.

---

## Data sources

The engine reads three CSVs from a directory. Nothing in `core/` knows or cares
where they came from, which is what makes the sources pluggable:

- **`data/generator.py`** — seeded synthetic batches, where each case type is
  constructed to defeat one specific shortcut, and ground truth is emitted
  alongside.
- **`integrations/razorpay.py`** — a real Razorpay settlement recon report
  (`GET /v1/settlements/recon/combined`) mapped into the same records. Amounts
  arrive as integer paise, so nothing rounds on the way in, and `settlement_id`
  aggregation reconstructs the payout that actually reaches the bank.

`integrations/` is the only module in the project permitted to open a socket, and
`core/` is forbidden from importing it. That asymmetry is deliberate: the data
source may reach the network, the matcher may never.

Razorpay cannot supply the bank statement, and that is not a limitation of the
adapter — it is the problem itself. If one system held both sides there would be
nothing to reconcile.

---

## What the cash layer knows that the engine does not

The engine sees an absent credit. It cannot tell whether that means *not yet
arrived* or *never arriving*, because on the evidence available those are the same
observation. `cash/position.py` makes that temporal judgement, because it knows
the settlement window: a settlement at T+1 with no credit is normal, and the same
settlement at T+9 is an incident.

Keeping that distinction out of the matcher is the same instinct as everything
above — each layer decides only what its evidence supports, and says so when it
cannot.

---

## Extension points

| you want to | change |
|---|---|
| reconcile a different bank's cycle | `core/config.py` — window and tolerances are per-counterparty properties |
| add a data source | a module under `integrations/`, emitting the three CSVs |
| add a matching rule | a tier in `core/matcher.py`, placed by evidence strength, not convenience |
| give the agent a new question | a read-only tool in `agent/tools.py` |
| swap model vendor | `agent/llm.py` — the shim exposes two operations |

The one extension the architecture will not accommodate is a tool that lets the
model change a reconciliation outcome. That is not an oversight to be fixed
later; it is the property the rest of the design is built to protect.
