# Engineering log

A running record of what actually broke during this build and what I did about
it. Written as it happened, not reconstructed at the end.

---

### 1. `UnicodeEncodeError` on the rupee sign — Day 1

**What broke.** The first smoke test of the money formatter crashed:

```
UnicodeEncodeError: 'charmap' codec can't encode character '₹'
```

Windows consoles default to the cp1252 code page, which has no code point for
U+20B9 (`₹`). Any `print()` of a formatted amount kills the process.

**Why it mattered more than it looked.** This is not a cosmetic bug. The entire
demo path — `run_demo.py` printing a metrics table — is built on formatted
currency. A judge cloning the repo on a stock Windows terminal would have hit a
traceback on the headline output, before seeing a single metric. It surfaced on
day 1 only because I ran the formatter through a console rather than trusting a
unit test that compares strings in memory.

**How I got out.** `_rupee_sign()` in `core/normalize.py` probes
`sys.stdout.encoding` and falls back to `Rs ` when U+20B9 is not encodable,
rather than assuming a UTF-8 terminal or forcing a reconfigure that would only
push the mojibake one layer down.

**What I changed in how I work.** Smoke-test through the real output channel,
not just through assertions. A string that compares equal in a test can still be
unprintable on the target machine.

---

### 2. The exception list was lying about *why* — Day 2

**What broke.** Not a crash. Two settlements contending for a single bank credit
were being reported as `no_candidate_found`, when the truth was the opposite:
there was a candidate, and it was wanted by two settlements at once.

I found it while writing the boundary tests, not while running the engine. The
counts had looked perfect — the row was correctly refused, correctly landed in
the exception list, correctly excluded from the match rate. Every headline
number was right. Only the *reason* was wrong.

**Why it mattered.** The entire pitch of this project is an honest exception
list. An exception list that says "nothing resembled this" when the real story
is "two settlements both wanted the same credit" sends a finance operator
looking for a missing payment that is sitting right there. The number was right
and the explanation was wrong, which is arguably worse than being visibly wrong
— it fails silently and confidently.

The cause was a control-flow gap. After constraint propagation reached its fixed
point, the leftover handler only treated `len(remaining) > 1` as ambiguous.
A settlement left with exactly one unclaimed candidate fell through to the
generic no-candidate tier, because reaching that state at all requires
contention — which the code never said out loud.

**How I got out.** Added `contested_candidate` as a distinct exception reason and
handled `len(remaining) == 1` explicitly, with a note naming the competing
credit. The propagation loop's fixed-point property is what makes the inference
sound, so I wrote that reasoning into a comment rather than leaving it implicit.

**What I changed in how I work.** Accuracy metrics cannot catch this class of bug
— a wrong label on a correctly-refused row scores identically to a right one. I
now treat the exception *taxonomy* as something to test directly, and the tests
assert `exception_reason`, not just `status`.

---

### 3. I nearly built the agent on an API shape that no longer exists — Day 3

**What broke.** Nothing, in the end — but only because I checked before writing.

My plan for getting structured JSON out of the model was assistant prefill: seed
the assistant turn with `{` so the response is forced into JSON. That is the
pattern I reached for automatically. On Claude Opus 5 it returns a **400**.
Prefill was removed across the Opus 4.6+ family. Had I written it from memory,
the reasoning layer would have failed on its very first live call — and since
the whole system is designed to degrade gracefully when the API is unavailable,
it would have failed *quietly*, degrading to placeholder explanations. I would
have seen "reasoning unavailable" and gone looking for a missing API key.

The correct current shape is `output_config={"format": {"type": "json_schema",
"schema": ...}}`, which constrains the response server-side and needs no prefill
at all. Two other priors were stale the same way: `budget_tokens` for extended
thinking is now rejected (it is `thinking={"type": "adaptive"}`), and the
`output_format` parameter has been superseded by `output_config.format`.

**Why it mattered.** My graceful-degradation design would have masked the bug.
Fallbacks that swallow failures are good for users and dangerous for developers:
they convert a loud error into a silent downgrade. The safety net I built to
handle missing keys would have hidden a bug that had nothing to do with keys.

**How I got out.** Read the current API reference before writing the client
rather than after debugging it. Cost: a few minutes. Would-be cost: an evening
chasing a phantom auth problem.

**What I changed in how I work.** Two things. First, I treat my recall of any
fast-moving API as a hypothesis, not a fact — LLM APIs in particular changed
shape several times in 2025-26. Second, I made the degradation path *loud about
which* failure it hit: `ReasoningStats` records `api_errors` separately from
`unavailable`, and the run prints them separately, so "no key" and "the call
failed" can never again look like the same thing on screen.

---

### 4. The held-out set worked — and then found a bug in my own benchmark — Day 4

**What broke.** Two things, in sequence. The second one is the interesting one.

**First: a real recall gap.** Dev scored 100% precision and 100% recall. The
held-out set — different seed, harder mix, sealed until thresholds were frozen —
came back at 100% precision but **84.2% recall**. Every miss was a split
settlement: 9 of 10 failed, against 8 of 8 on dev.

The cause was not a bug. The held-out batch is generated with `max_lag_days=4`,
modelling a bank on a slower settlement cycle, while the engine is configured for
T+2. Legs landing on day three and four fall outside the window, so the split
cannot be assembled. The engine's response was to refuse rather than mis-bind —
precision stayed at 100%. The conservative failure mode behaved exactly as
designed.

The tempting fix was to widen the window to 4 and watch the number go green.
That is tuning on the evaluation set, so I did not do it. `core/config.py` has
not been modified since the first commit, which is checkable in git history.
Instead I wrote `evaluation/sensitivity.py` to measure the trade — and that is
where the second problem surfaced.

**Second: my ground truth was lying at the edges.** The sweep reported that
*tightening* the amount tolerance to zero produced **four false positives** on
dev. That is backwards on its face: a stricter threshold should never invent
matches. I went to look at the four cases expecting an engine bug in tier
cascade fallthrough.

They were not engine bugs. For all four, `expected == predicted` — the engine had
bound each adversarial twin to its *correct* counterpart. The evaluator was
scoring four correct matches as false positives.

The reason: I had generated the "ambiguous" twins two paise apart. At the shipped
50-paise tolerance they are genuinely undecidable, so the label was right. At
zero tolerance each twin has exactly one exact-amount candidate, so they become
*perfectly* decidable — and my ground truth, which hard-codes
`expected_resolution: exception_ambiguous`, kept insisting they were not. I had
encoded a **threshold-dependent property as an absolute label**.

**Why it mattered.** This is the failure mode that quietly invalidates a whole
evaluation. Every headline number I had reported was correct *at the shipped
configuration*, so nothing looked wrong. But the benchmark could not be trusted
off that one operating point — which is precisely what a sensitivity analysis is
for, and precisely the question a panel would ask. Worse, it made my central
adversarial claim weaker than I thought: I was testing "can the engine refuse
when two things are two paise apart", not "can it refuse when two things are
genuinely indistinguishable".

**How I got out.** Regenerated the adversarial twins with **identical** nets on
the same date and no UTR on either bank row. Now no threshold anywhere can
separate them, so "refuse both" is correct at every configuration — and the sweep
confirms it: zero conflations across every tolerance from 0p to Rs 50.
`test_ambiguous_twins_are_undecidable_at_any_tolerance` locks it in so the
weakness cannot creep back.

I changed the *test*, not the engine, and the change made the benchmark harder
rather than easier. The headline numbers were unaffected.

**What I changed in how I work.** A benchmark is code, and it deserves the same
suspicion as the code under test. I had already written tests asserting my
generator produced what it claimed — but every one of them checked the data
against the *shipped* thresholds, so they could never have caught a label that
was only conditionally true. Now I sweep the parameter space partly to find
answers and partly to interrogate the harness, on the principle that a result
which makes no sense is usually the measurement failing, not the thing measured.

---

### 5. The same stale-API trap, one provider over — Day 5

**What broke.** Nothing, again, and for the same reason as entry 3 — which is
the point of recording it.

The build switched from an Anthropic key to an OpenAI one. My hands started
writing `client.chat.completions.create(...)` with
`response_format={"type": "json_schema", ...}`, because that is the shape I know.
Current OpenAI structured output goes through the **Responses API**:
`client.responses.create(..., text={"format": {"type": "json_schema", "name":
..., "strict": True, "schema": ...}})`. The model IDs I would have reached for
were stale too; the current family is `gpt-5.6-sol` / `-terra` / `-luna`.

Entry 3 was the identical mistake with Anthropic prefill. Two providers, two
stale priors, one week apart. The rule I wrote down after the first one - treat
recall of a fast-moving API as a hypothesis - is the only reason this cost ten
minutes of reading rather than an evening of debugging.

**What it confirmed about the design.** The swap touched three files and no
matching code. `core/` cannot import a vendor SDK - there is a test that parses
its AST and fails the build otherwise - so the blast radius was structurally
bounded before I started. I replaced the direct client with `agent/llm.py`, a
shim exposing exactly two operations, and added a test asserting reconciliation
output is byte-identical under Anthropic, OpenAI, and no provider at all.

That test should be trivially true, since the engine never consults the shim.
Asserting it anyway is the cheapest possible insurance on the central claim of
the whole project: the model explains, it does not decide. A provider swap is
the most direct way that claim could quietly become false, so now it cannot.

**What I changed in how I work.** Nothing new - entry 3's rule held. Worth
noting that the payoff was not avoiding one bug. It was that "which model
vendor" turned out to be a genuinely small decision, because the architecture
had already made it one.

---

### 6. My own safety net poisoned the cache — Day 5

**What broke.** The first run with a real API key made zero API calls. Every
exception came back with `source: cached_trace` and a hypothesis reading
*"No agent explanation available (API error: 401 ... sk-fake-...)"*.

Earlier I had tested graceful degradation by running with a deliberately fake
`OPENAI_API_KEY`. That worked - the run completed, the errors were reported, the
metrics were untouched. What I did not notice is that the placeholder
investigations it produced were written to the **persistent trace cache**, which
is committed to the repository. Thirty-one entries, each recording an absence of
reasoning, stored as though they were reasoning.

**Why it mattered.** Three reasons, escalating.

First, it silently blocked the real run: cache hits meant the working key was
never used.

Second, the poisoned entries came back labelled `source: cached_trace`, which is
exactly what a legitimate replayed answer looks like. There was no signal
distinguishing "we replayed a real investigation" from "we replayed a failure".

Third, and worst: that cache file is the thing a judge sees. Had I not looked at
the output closely, I would have committed thirty-one API errors into the
repository as the project's demonstration of agent reasoning, and the keyless
demo - the whole point of committing traces - would have shown 401s.

The cause is a one-line oversight with a wide blast radius. `_call` returns
placeholder dicts on API failure so the run can continue. Those placeholders have
the same shape as real investigations, so the caching path could not tell them
apart and stored them like anything else. My degradation mechanism and my caching
mechanism were each correct alone and wrong together.

**How I got out.** Placeholders now carry an `_unavailable` sentinel and
`is_placeholder()`. They are never written to the cache, and a cached placeholder
is treated as a **miss** rather than a hit - so a cache already poisoned by a
failed run heals itself on the next good one instead of needing manual purging.
Two regression tests cover both halves.

While fixing it I found a second, quieter version of the same class of bug: the
cache key was a hash of the evidence only, not of the system prompt. Editing the
prompt would appear to take effect while every existing case silently replayed an
answer written under the old instructions - undetectable, because a cached trace
is indistinguishable from a fresh one once stored. The key now folds in a hash of
the instructions.

**What I changed in how I work.** I had tested the failure path and I had tested
the caching path. I had not tested the failure path *followed by* the caching
path. The bug lived in the seam, which is where this kind of bug always lives.
Now I ask what a fallback leaves behind, not just whether it fires - because a
fallback that persists its own output is no longer a fallback, it is a writer.
