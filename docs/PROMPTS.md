# Claude Code prompt sequence

Copy-paste, one at a time. Do not batch. Wait for tests to pass before the next.

## Session hygiene

Open every session with:

```
Read CLAUDE.md, docs/ROADMAP.md and docs/DECISIONS.md.
Tell me which phase we're in, what's done, and what the next single module is.
Don't write code yet.
```

Close every session with:

```
Summarise what changed, what's tested, and what's not.
Add any non-obvious choices to docs/DECISIONS.md as ADR entries.
Add anything that didn't work to docs/NEGATIVE_RESULTS.md.
List the three things most likely to be wrong in what we just wrote.
```

---

## Phase 0

```
Clone int-brain-lab/ibl-ai-agent into external/. Don't run its agent.
Read its data documentation and tell me exactly what the compressed BWM
representation contains: fields, units, time resolution, which behavioural
traces, and total size on disk. Write this into docs/PRIOR_ART.md under
section B, replacing my [A] markers with verified facts.
```

```
Clone yzhang511/NEDS into external/. Read the repo — don't run training yet.
Report: how sessions are loaded, how spikes are tokenized, exactly what goes
into the session embedding, what train_eids.txt/test_eids.txt contain, what
the decoder heads output, and whether anything in the codebase produces a
calibrated probability or an uncertainty estimate. Be specific about file
paths. I want to know precisely what I'd be reimplementing.
```

```
Set up the NEDS environment and reproduce one decoding number from the paper
on a single session. Report what broke and what you had to change. If it
can't be reproduced in a reasonable time, say so and stop — don't fabricate a
partial run.
```

```
Find the SpikeProphecy benchmark (arXiv 2605.12992). Read it and tell me
whether it already covers autoregressive neural population forecasting on
IBL repeated site in a way that makes Phase 9 of our roadmap redundant.
Give me a yes/no recommendation on cutting Phase 9, with reasoning.
```

```
Scaffold the repo per CLAUDE.md section 9. pyproject.toml with pinned deps
(only the "fixed" list from section 8), ruff + black pre-commit, pytest,
GitHub Actions running tests on push. Create docs/DECISIONS.md and
docs/NEGATIVE_RESULTS.md with templates. Empty test suite that passes.
Nothing else — no placeholder modules with stub functions.
```

## Phase 1

```
Implement neurodecoder/data/manifest.py. Build a session table from the ONE
API with the columns listed in ROADMAP Phase 1. Persist as parquet. Include
a test that the manifest has no null subject IDs and that unit counts are
positive integers. Handle credential setup by reading from env vars — never
hardcode. Don't implement the other backends yet.
```

```
Implement the three backends behind the load_session interface in ROADMAP
Phase 1. Then write tests/test_backend_agreement.py: for 5 eids, assert the
three backends return identical unit counts and identical total spike counts.
Run it. If they disagree, do NOT add a tolerance — investigate and report the
cause to me before changing anything.
```

```
Implement data/cache.py, content-addressed on (eid, PREPROC_VERSION,
config_hash). Test that a cache hit returns arrays identical to a cold load,
and that bumping PREPROC_VERSION invalidates.
```

## Phase 2

```
Read docs/SPLITS_AND_LEAKAGE.md in full. Implement splits/registry.py and
splits/guards.py exactly as specified there — all six split types, JSON
serialization with a hash, and assert_split_valid raising on every listed
condition. Write tests that each guard actually fires: construct a
deliberately leaky split for each rule and assert it raises. This module is
the backbone of the project's scientific validity; over-test it.
```

```
Implement preprocess/binning.py and preprocess/normalize.py. normalize must
take a split object, not a session — there should be no API path that lets
someone normalize without declaring a split. Put the shapes in the docstrings
in (n_units, n_bins) form and assert them at function boundaries.
```

```
Implement targets/ for wheel velocity, choice, movement onset, block, and a
discretized movement_state. For wheel velocity, propose 2-3 filtering
approaches with their trade-offs, pick one, and write it into
docs/DECISIONS.md as an ADR. Do not tune the filter against any metric.
```

## Phase 3

```
Implement evaluation/contract.py implementing the six-row table in CLAUDE.md
section 5, and evaluation/nulls.py with the circular-shift and
trial-structure-only nulls. The API should make it awkward to evaluate a
model without producing the full table.
```

```
Implement models/baselines/ — ridge for wheel velocity, logistic for choice
and movement_state, Poisson GLM, and multi-session reduced-rank regression.
Run the full evaluation contract on a within_session split for 10 sessions.
Show me the six-row table. Do not interpret the numbers yet — just show them.
```

```
Our baselines beat null_trialstruct by [X]. Before I believe it: audit the
pipeline for leakage against every route listed in
docs/SPLITS_AND_LEAKAGE.md. Check each one explicitly and report pass/fail.
Assume the result is wrong until you've ruled them out.
```

```
Implement nwb/probe.py — thin schema inspection only, no normalization, no
model. Run it against three dandiset 000409 processed files and five NWB
files from unrelated dandisets. Produce a table of what it found and what it
couldn't interpret. Tell me how much work full NWB intake looks like.
```

## Phase 4

```
Run the full evaluation contract for every baseline across all six split
types. Produce the generalization gap table and a plot. Then tell me
honestly: is the cross-animal drop large enough to justify this project's
premise? If it's small, say so — that's important information.
```

## Phase 5

```
Implement representation/set_encoder.py following the architecture in
ROADMAP Phase 5. Reuse NEDS's tokenization approach — read
external/NEDS first and tell me what you're taking. Write
test_permutation_invariance, test_variable_unit_count and test_masking
BEFORE the implementation.
```

```
Train the encoder on a held_out_animal split with 20 training animals.
Compare against baseline_ridge using the evaluation contract. If it doesn't
beat ridge on held-out animals, do not increase capacity — first check the
split, then the normalization, then the session embedding handling at test
time. Report which you checked.
```

## Phase 6

```
Implement uncertainty/ with temperature scaling, a 5-member deep ensemble,
MC dropout, and split conformal (sets for classification, conformalized
quantile regression for wheel velocity). The calibration split must be
disjoint at animal level from both train and test — assert this.
```

```
Produce reliability diagrams and coverage-vs-nominal plots for every method
across every split type. I expect conformal coverage to break under animal
shift because exchangeability fails. Report it plainly if it does — that's a
finding, not a bug, and it goes in the paper.
```

## Phase 7

```
Implement ood/scores.py with Mahalanobis, k-NN, energy, ensemble
disagreement, and metadata-space distance. Then run the defining experiment
from ROADMAP Phase 7: per-session OOD score vs per-session decoding error on
held-out animals, Spearman correlation, plus risk-coverage curves for
abstention by OOD score vs by softmax confidence vs random, with AURC.
```

```
Before I believe the OOD result: check whether the OOD score is just
measuring unit count, or session duration, or firing rate. Partial out each
confound and report the correlation that survives.
```

## Phase 8

```
Implement cli/analyze.py and the artifact writers. Then implement
tests/test_agent_no_invention.py: parse the generated HTML report, extract
every numeral, and assert each appears in predictions.parquet,
uncertainty.parquet or metrics.json. Write the test first.
```

```
Implement agent/. It may only read artifacts from models/ and evaluation/.
It must have no numerical tools and no data access. Show me the import graph
to prove it.
```

---

## Prompts to use when something feels wrong

```
This number is better than I expected. Before we move on, argue the case that
it's leakage. Be specific about which mechanism.
```

```
You changed something in preprocess/ or splits/ this session. Show me the
diff and justify each line against CLAUDE.md rule R6.
```

```
We're three phases past where this should have been caught. Re-run Phase 3's
evaluation contract with today's code and tell me if anything drifted.
```

```
I want to add [X]. Tell me why I shouldn't, against the roadmap and the scope
limits in CLAUDE.md section 2. Then tell me what it would cost.
```
