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
