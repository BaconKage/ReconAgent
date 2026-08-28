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

---

### 7. The README's own command was broken — Day 6

**What broke.** The fresh-clone verification, following my own README in order:

```
1. python run_demo.py                      OK
2. python run_demo.py --dataset holdout    OK
3. python run_demo.py --ask "why didn't pay_f7atwyam1n reconcile?"
   -> "I could not find a transaction ID in that question..."
```

Audit trails are written per run, and the Q&A layer reads the most recent one.
Step 2 made the holdout run the most recent, so a dev transaction was genuinely
absent from it. The answer was technically correct about that trail and useless
to the person asking, who had reconciled that transaction ninety seconds earlier.

**Why it mattered.** Not the severity - it is a small bug. What matters is how it
was found. 155 tests passed. The Q&A tests passed, including one asserting that
an unknown ID returns "not found", which is the very behaviour that was wrong
here. Every test built its own trail in a temp directory and asked about that
trail, so none of them could ever encounter two runs in sequence. The bug lived
in the gap between "each command works" and "the commands work in the order the
README gives them".

It would have been a bad thirty seconds of demo video: run the tool, run it on
the held-out set to show the honest numbers, then ask it a question and have it
say it has never heard of the transaction.

**How I got out.** The Q&A layer now falls back to searching every trail in the
directory and names the run an answer came from. An ID that exists in no run
still returns "not found" - that behaviour was correct and is still tested.

**What I changed in how I work.** I had verified that every command works. I had
not verified that they work *in sequence*, from a clean clone, in the order I
tell someone to run them. Those are different claims, and only the second one is
what a judge actually experiences. The fresh-clone rehearsal is not a formality
at the end of the build; it is the only test that exercises the product rather
than the code.

---

### 8. A convenience accessor that silently produced invalid JSON — Day 7

**What broke.** The very first turn of the very first investigation:

```
TURN 1: step failed: Extra data: line 1 column 320 (char 319)
```

The model had returned this, twice, concatenated:

```
{"thought":"...","action":"credits_near_settlement",...}{"thought":"...","action":"credits_near_settlement",...}
```

Two complete, valid, identical JSON objects glued together - which is not valid
JSON. `json.loads` gave up at the boundary.

**The cause.** I was reading the response with `response.output_text`, the
documented convenience accessor on the OpenAI Responses API. Inspecting the raw
response showed five output items:

```
[0] reasoning
[1] message   -> 326 chars of JSON
[2] reasoning
[3] reasoning
[4] message   -> the same 326 chars
```

`output_text` concatenates the text of **every** message item. A reasoning model
routinely emits more than one, interleaved with its reasoning items. For prose
output the result is merely repetitive; for a schema-constrained JSON response it
is a parse error every time two messages appear.

**Why it mattered more than the fix suggests.** This is the third entry in this
log about trusting a familiar-looking API surface, and it is the most insidious
of them, because `output_text` is not deprecated, not misused, and not wrong -
it does exactly what it documents. It is simply the wrong tool for structured
output, and nothing about the call site says so.

It is also *intermittent*. Whether a second message appears depends on the
prompt, so my earlier batched calls had worked fine for over a hundred traces.
Had I shipped the investigation loop without testing a single case end to end, it
would have failed on some cases and not others, and the failures would have been
absorbed by my own graceful-degradation path into "escalate to human" - which is
exactly what an unresolvable case looks like. Silent, plausible, and wrong.

**How I got out.** `_openai_text` now walks the output items and returns the
**last** message rather than the concatenation of all of them, joining content
parts only within that single message. Added `parse_first_object` as a second
line of defence on both the batched and the investigation paths, and a test that
feeds the shim a synthetic two-message response.

**What I changed in how I work.** For anything whose output feeds a parser, I now
look at the raw response shape once before trusting an accessor - `len(r.output)`
and the item types cost one print statement. Convenience accessors are designed
for display, and display tolerates concatenation. Parsers do not.

---

### 9. My headline metric could not measure the thing it claimed — Day 7

**What broke.** Not the code. The evidence.

I had been reporting **100% precision, 0 false positives** on both datasets, and
treating that as the project's strongest claim. Scaling the benchmark - which I
expected to be a throughput exercise - produced this, across two independent
seeds:

| rows | settlements | false positives | FP rate |
|---|---|---|---|
| 255 | 87 | 0 / 0 | 0.00% |
| 1,008 | 346 | 0 / 0 | 0.00% |
| 5,042 | 1,727 | 10 / 6 | 0.58% / 0.35% |
| 24,967 | 8,547 | 46 / 35 | 0.54% / 0.41% |

The rate is roughly **constant** at half a percent of settlements. It is not that
the matcher degrades at scale; it is that the error was always there and my test
set could not see it. 0.5% of 87 settlements is 0.44 expected errors, so observing
zero is the single most likely outcome *even if the true rate is exactly 0.5%*.

I had run a test with no power to detect the thing it was testing, got the answer
I wanted, and put it at the top of the README in bold.

**Why it mattered.** This is the most uncomfortable entry in this log, because
nothing failed. Every test passed. The number was arithmetically correct. The
dev and holdout figures are still true statements about those batches. What was
wrong was the *inference* - treating "we observed zero" as "the rate is zero",
when the sample could not distinguish those cases.

It is also the failure mode I had already written a test against in spirit. In
entry 4 I fixed a benchmark whose labels were only conditionally true. Here the
labels were fine and the *sample size* was the problem, which no amount of
staring at the ground truth would have revealed. Only more data did.

**What actually causes them.** Every false positive comes from tier 3
(`amount_date_window`); none from splits, none from the ambiguity guard. A
settlement that should be a split, or should have no counterpart at all, gets
bound to a single unrelated credit that happens to fall within 50 paise and the
three-day window. The ambiguity guard triggers on two or more candidates and has
no answer to exactly one coincidental candidate. At 87 settlements you will
almost never see it. At 8,547 you see it 46 times.

The adversarial claim, by contrast, held up completely: 0 conflations out of 495
near-duplicate pairs at 25,000 rows. That one *was* adequately powered, and it is
the claim the refusal design was actually built to support.

**How I got out.** I did not fix it. With the deadline close I judged that
reporting it accurately was worth more than a partial fix I could not properly
verify, so the README now leads the results table with a warning that those two
sets cannot measure a false-positive rate, carries the measured ~0.5% figure and
its cause, and states that the fix is unimplemented. `benchmark.py` is committed
so anyone can reproduce all of it. The throughput claim, which turned out to be a
50x overstatement measured over 1.8 milliseconds of interpreter noise, is
corrected in the same place.

**What I changed in how I work.** Before quoting a rate, I now ask what the
smallest rate that sample could have detected is. If the answer is larger than the
number I am about to report, I do not have a measurement - I have an absence of
evidence, and those are not the same thing. A clean 100% on a small sample should
increase suspicion, not confidence.


---

### 10. Fixing the false positives, and the idea that died first - Day 8

**What I tried first, and why it failed.** The obvious fix for the split-related
false positives was: before accepting a lone amount+date match, check whether a
split decomposition could also explain the settlement. If two stories fit, refuse.

I measured it before building it. A split decomposition exists for **100% of true
matches and 100% of false ones** at 25,000 records. Useless as a discriminator -
and it confirms, harder than I expected, the warning I had written into
`splits.py` many commits earlier: given enough candidates, *something* always sums
to the target. Two minutes of measurement killed a fix I would otherwise have
spent an afternoon building.

**What actually separates them.** Comparing true against false tier-3 matches:

| | median delta | others within Rs 50 |
|---|---|---|
| true | 0 paise | 3 |
| false | 28 paise | 16 |

Both are informative and neither is sufficient. True matches sit at delta zero;
coincidental deltas spread evenly across the tolerance band, so their median lands
near half of it. And false positives happen in neighbourhoods five times more
crowded.

Density alone traded 3.4 true matches per false positive caught - not worth it.
Adding lag as a third factor took that to **1.0:1**, which is worth taking, since binding
money wrongly costs far more than asking a human to look.

**The trap I nearly walked into.** I found the best threshold by sweeping it
against the benchmark, which is tuning on the evaluation set - the exact thing
entry 4 was about. So I rounded the swept 0.046 to a plain 0.05 and validated on a
seed the guard had never seen. Seed 99 gives 35 -> 22 false positives against seed
7 at 46 -> 24. It generalises.

**What I did not do.** Chase the remaining half. Those are cases where a
coincidental credit is genuinely indistinguishable from a real one on amount and
date; no feature I have separates them, and with 52 true matches per false one,
any aggressive rule costs more than it saves. The honest fix needs signal the data
does not carry - narration text, customer names in the bank reference, historical
pairing. That is stated in the README rather than papered over.

**What I changed in how I work.** Measure the discriminator before building the
fix. Both my candidate features were plausible, one was worthless, and finding
that out cost two minutes instead of an afternoon. And when a threshold has to be
swept, round it and validate on data it has never seen - a number quoted to three
decimal places is usually a number fitted to the test set.


---

### 11. A 10x speedup, and the one line of trace that nearly went with it - Day 8

**What broke.** Nothing, but the throughput number in my README was fiction, and
the profile said why: `_amount_ok` called 21.5 million times on a 25,000-row
batch. Tiers 2 through 5 each scanned every unclaimed credit for every
settlement, and `_unclaimed()` rebuilt the entire list 6,229 times in one run.
Matching was quadratic and I had been quoting 141,000 rows/sec, a figure measured
over 1.8 milliseconds of interpreter noise.

**The rule I set before touching anything.** An optimisation must produce
byte-identical output or it is a bug, not a speedup. So I captured a full
signature first - status, match type, confidence, exception reason, linked IDs,
the complete rule trace and every near miss - across dev, holdout and two
generated batches. 12,940 groups. Then I compared after every change.

**The near miss.** The first indexed version came back identical on three of four
datasets. Holdout differed on **one line of one rule trace** out of 12,940 groups:

```
old: tier4 split: no subset of credits in the window sums to net
new: (line absent)
```

No status changed. No confidence, no link, no metric - precision, recall and the
exception counts were all identical. Purely a line of explanation, on a single
settlement.

The cause was that I had pre-filtered the split pool by minimum leg size, which
was marginally faster and dropped the pool below the two-entry threshold that
decides whether tier 4 writes its "nothing summed to net" line at all. I reverted
it. A reviewer opening that transaction would have seen an engine that never
considered a split, when in fact it had. The audit trail is a product feature
here, not debug output, and a faster engine that quietly edits its own explanation
of itself is not a faster engine.

Had I compared only statuses and metrics - the obvious thing to compare - I would
never have seen it.

**What actually made it fast.** Three changes, each verified separately:

| | |
|---|---|
| `core/index.py` | credits bucketed by date, sorted by amount, two binary searches per lookup |
| tier 2 | prefix-match truncated UTRs by bisect, not by scanning 17,000 UTRs each time |
| subset-sum | skip the too-large prefix by bisect instead of walking past it |

The third was the surprise: at 100,000 rows the split search was **71% of total
time**, and most of that was the DFS stepping over legs it was going to reject.

| rows | before | after |
|---|---|---|
| 25,000 | 9.18 s | 0.85 s |
| 100,000 | ~147 s (extrapolated) | 6.48 s |

**What I did not fix.** Beyond 50,000 rows it is still mildly superlinear, and the
residual is subset-sum itself: a denser batch puts more credits inside each
settlement's window. Capping the pool would fix it and would also cut the
coincidental splits that grow with density - but it changes results, so it belongs
in its own before-and-after rather than being slipped into a speed change.

**What I changed in how I work.** When verifying that an optimisation is
behaviour-preserving, compare everything the system emits, not just the fields
that feed the metrics. The metrics were identical in the version I nearly shipped.


---

### 12. Three claims a reviewer could have disproved faster than I could defend - Day 9

**What broke.** Nothing in the engine. Three things in the *evidence*, which on
this project is the product.

**First: I invited a check that failed.** The README said, twice, that
`core/config.py` had not changed since the first commit, and told the reader it
was provable from git history. It was not:

```
$ git log --oneline -- core/config.py
9f520bd Coincidence guard: halve the tier-3 false positives
9c41d6d Phases 1-3: synthetic data, deterministic engine, reasoning layer
```

The coincidence guard appended two new parameters. It altered no existing
threshold, so the claim I *meant* - that nothing was tuned against the held-out
set - was true the whole time. But the sentence I actually wrote was checkable in
ten seconds and false, on the one page where being checkable is the point. Anyone
who ran that command would have been right to distrust every other number.

The fix was to say the true thing, which is stronger: no *matching* threshold has
changed, config was touched exactly once, here is the commit, diff it.

**Second: my grounding number was not reproducible.** "Across 104 traces the
model wrote 264 record IDs and zero were ungrounded" was in bold, and nothing in
the repository computed it. Counting the committed traces every plausible way
gives 183 cited IDs, 199 unique per trace, 310 including prose, 752 across the
whole blob. Never 264. I no longer know where the figure came from - most likely
a run whose traces were later regenerated.

This is entry 9's mistake wearing different clothes. There, I quoted a rate my
sample could not measure. Here, I quoted a count nothing recomputed. Both are the
same failure: an assertion that had stopped being attached to a measurement, in a
document whose entire argument is that its assertions are attached to
measurements.

So I wrote `verify_grounding.py`, which rebuilds each case's evidence bundle
through the real `build_evidence_bundle`, and checks every ID the model wrote
against what it was actually shown - including tool-call *arguments*, which the
old figure did not cover. A model that invents a bank row and then queries a tool
about it has hallucinated, even if the tool returns nothing and the invented ID
never reaches the answer. That is where it would surface first.

The honest number is **395 IDs, 0 ungrounded** - larger than the claim it
replaces, under a stricter definition, and now reproducible by one command that
exits non-zero if it ever stops being true.

**Third: I retracted a number in the README and kept printing it.** Entry 9
established that the throughput headline was interpreter noise measured over
1.8 ms. The README said so. `run_demo.py` went on printing `120,563 rows/sec` off
a 2 ms run, `app.py` put a rate in the sidebar, and `benchmark.py` labelled the
250 -> 5,000 step **quadratic** - a verdict produced by dividing by a
sub-millisecond baseline, which is the exact error the script exists to correct.

A judge who read the README saw the retraction. A judge who ran the code saw the
fiction. The demo was quietly contradicting the confession.

All three now refuse to quote a rate below a 50 ms floor and say why. The
benchmark declines to classify scaling against an unmeasurable baseline, which is
the same discipline the engine applies to an ambiguous match: the measurement
does not support a verdict, so it does not render one.

**What I changed in how I work.** A retraction is not finished when the prose
changes. It is finished when nothing in the system still emits the retracted
claim - and the place to look is the output a reviewer sees before they read
anything. More generally: every number I put in bold now needs a command that
regenerates it. If I cannot write that command, the number is a memory, not a
measurement, and it should not be in bold.
