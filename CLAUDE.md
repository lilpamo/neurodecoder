# CLAUDE.md — neurodecoder

Project constitution. Read this file in full at the start of every session.
If anything below conflicts with a request in chat, say so before acting.

---

## 1. What this project is

A **reliability layer for cross-animal neural population decoding**, built on IBL
Neuropixels data, with an NWB intake path for arbitrary files.

The scientific question is *not* "can we decode behaviour from spikes" — that is
settled. The question is:

> When a decoder trained on many animals is applied to a new animal or a new
> recording configuration, can we tell in advance whether its prediction should
> be trusted — and does acting on that signal measurably reduce error?

Everything else in the repo exists to make that question answerable.

## 2. What this project is NOT

Do not let the codebase drift toward any of these without an explicit decision
recorded in `docs/DECISIONS.md`:

- A foundation model for neural data. NEDS, POYO+ and NDT2 exist. We are not
  competing on decoding accuracy at scale.
- A spike-sorting pipeline. We consume sorted units only.
- An LLM that predicts behaviour. See §6.
- A hosted web app. Analysis is CLI first. The one exception is the Phase 8b
  local app, a browser UI on the user's own machine that only calls CLI entry
  points; see "Local app for non-programmers (Phase 8b)" in `docs/DECISIONS.md`.
- A general neuroscience analysis library. SpikeLab exists; prefer wrapping it
  over reimplementing it.

## 3. Non-negotiable scientific rules

These are enforced by tests. Do not weaken a test to make code pass; fix the code
or escalate in chat.

**R1 — No split without a registry.** Every train/val/test partition is produced
by `neurodecoder.splits` and serialized to a split file with a hash. Models never
receive raw session lists. See `docs/SPLITS_AND_LEAKAGE.md`.

**R2 — No random time-point splits.** Ever. Not for a quick check, not for a
smoke test. Temporal blocks or whole trials, with a gap. A "quick sanity run"
that leaks is worse than no run.

**R3 — Normalization statistics are fit on training data only**, per the split
definition, and stored in the model artifact. If you find yourself computing a
z-score over a whole session before splitting, stop.

**R4 — Every reported number ships with its baseline and its null.** A decoding
accuracy with no shuffle control and no trial-structure-only baseline is not a
result. See §5.

**R5 — Latent states are model outputs, not biological claims.** Name them
`latent_state_k`, never `engagement` or `preparation`, anywhere in code, plots,
or reports. Interpretation belongs in prose written by a human.

**R6 — Preprocessing is versioned and deterministic.** Changing bin width,
QC thresholds, unit filtering, or alignment bumps `PREPROC_VERSION` and
invalidates caches. Never change these silently to improve a metric.

**R7 — Seeds are explicit and recorded.** No implicit global RNG.

## 4. Before writing code in any session

1. Read `docs/ROADMAP.md` and identify which phase we are in.
2. Read `docs/DECISIONS.md` for decisions already made and
   `docs/NEGATIVE_RESULTS.md` for what has already been tried and failed.
3. State, in chat, the one module you are about to touch and what test will prove
   it works.
4. Do not touch modules outside that scope. If a change requires it, stop and say
   so.

## 5. The evaluation contract

Any model evaluation must report, in this order:

| Row | What |
|---|---|
| `null_shuffle` | Targets circularly shifted within session; destroys neural-behaviour alignment, preserves autocorrelation |
| `null_trialstruct` | Prediction from trial time / task variables only, no spikes |
| `baseline_ridge` | Ridge or logistic on binned counts, same split |
| `baseline_rrr` | Reduced-rank regression, multi-session |
| `model` | The thing being tested |
| `ceiling_within` | Same model, within-session split — the upper bound |

A result that does not beat `null_trialstruct` is not decoding. A result that
does not beat `baseline_ridge` does not justify a deep model. Say this plainly in
the output rather than burying it.

Metrics: for classification, balanced accuracy + AUROC + log loss + ECE. For
regression, R² + correlation, reported per-session, never pooled across sessions
without also showing the per-session distribution. Pooled R² across sessions with
different variance is misleading; the test suite checks for it.

## 6. The LLM boundary

Hard architectural rule.

```
NWB / ONE  →  deterministic preprocessing  →  trained model  →  numbers on disk
                                                                     ↓
                                                        agent reads the numbers
                                                                     ↓
                                                        prose, plots, report
```

The agent layer (`neurodecoder/agent/`) may only read artifacts produced by
`neurodecoder/models/` and `neurodecoder/evaluation/`. It has no access to raw
data and no numerical tools that could produce a prediction.

Every number appearing in a generated report must be traceable to a field in
`predictions.parquet`, `uncertainty.parquet`, or `metrics.json`. There is a test
(`tests/test_agent_no_invention.py`) that parses generated reports and asserts
every numeral appears in the source artifacts. Do not disable it.

## 7. Working style for Claude Code

**Do:**
- One module per session. Write the test first when the module has a clear
  contract.
- Put array shapes in docstrings, in the form `(n_units, n_bins)`, and assert
  them at function boundaries.
- Keep every tunable in a YAML config under `configs/`. No magic numbers in
  model code.
- Log every run with its config hash, split hash, preproc version, git SHA, and
  seed into `runs/<run_id>/manifest.json`.
- When a number surprises you, investigate before celebrating. Unexpectedly good
  decoding is almost always leakage.
- Write down failed experiments in `docs/NEGATIVE_RESULTS.md`. They are the most
  valuable thing in this repo and the easiest to lose.
- Every capability is a CLI entry point with structured JSON output; errors and
  refusals are written for non-programmers.

**Do not:**
- Refactor validated preprocessing code while implementing something else.
- Add a dependency without adding a line to `docs/DECISIONS.md` saying why.
- Increase model capacity to fix a generalization gap before checking splits.
- Silently handle a missing NWB field with a default. Missing means missing;
  propagate it to the capability report.
- Write more than ~400 lines before running anything.
- Generate synthetic "example" data that could be mistaken for real results.

## 8. Stack

Fixed: Python 3.11, PyTorch, NumPy, SciPy, scikit-learn, pandas, pyarrow, h5py,
pynwb, ONE-api / ibllib, PyYAML, pytest, matplotlib.

Added only when the phase needs it: `dandi` + `remfile` (Phase 1 streaming),
`spikeinterface` (NWB units reading), `polars` (only if pandas is measurably the
bottleneck), `netcal` or equivalent (Phase 6 calibration), `hydra` (only if
config composition becomes painful).

Deliberately excluded until justified: Dask, Zarr, xarray, MLflow, W&B, napari,
Plotly, Lightning. A `runs/` directory with JSON manifests is sufficient
experiment tracking for one person. Revisit at Phase 9.

## 9. Repository layout

```
neurodecoder/
  data/            session manifest, ONE + DANDI access, caching
  nwb/             intake, schema probe, capability report
  qc/              unit filtering, session-level QC
  preprocess/      binning, alignment, normalization  [VERSIONED — see R6]
  targets/         behavioural target construction
  splits/          split registry, leakage guards       [see R1, R2]
  representation/  population encoders (set-based, metadata tokens)
  models/          baselines/, deep/, heads/
  uncertainty/     calibration, conformal, ensembles
  ood/             session-similarity, gating, selective prediction
  latent/          state models  [descriptive only — see R5]
  evaluation/      metrics, protocols, the evaluation contract
  viz/             plots
  agent/           report generation  [LLM boundary — see §6]
  cli/
configs/
docs/
tests/
runs/              gitignored
```

## 10. Escalate to the human when

- A metric jumps more than you expected.
- A split produces fewer sessions than the config implies.
- An NWB file parses but yields zero units or misaligned timestamps.
- You are about to change anything in `preprocess/` or `splits/`.
- A phase's success criterion is not met and the fix would take more than a
  session.
- You believe a phase in the roadmap is the wrong next step. Say so.
