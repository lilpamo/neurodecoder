# Negative Results

Failed experiments and dead ends. The most valuable and easiest-to-lose record
in this repo. One entry per experiment, newest first.

---

### 2026-09-29 — Phase 3 first six-row table: decision gate not passed

**Run:** `runs/20260929T070943Z_phase3_first_table`, git SHA `81edcfe` (clean),
config `configs/runs/phase3_first_table.yaml`.

**Verdict lines (model vs null_trialstruct), as the contract printed them:**
- choice: model does NOT beat null_trialstruct (median Δauroc -0.071, wins 4/10 sessions, Wilcoxon p = 0.968): this is not decoding.
- block: model does NOT beat null_trialstruct (median Δauroc -0.307, wins 0/9 sessions, Wilcoxon p = 1): this is not decoding.
- wheel_velocity: model does NOT beat null_trialstruct (median Δr2 +0.053, wins 6/10 sessions, Wilcoxon p = 0.577): this is not decoding.
- movement_state: model beats null_trialstruct (median Δauroc +0.281, wins 10/10 sessions, Wilcoxon p = 0.000977).

The full six-row tables and the other verdict lines are in each target's
`report.txt` in the run directory.

**Status:** gate not passed; investigation pending.

### 2026-09-27 — Reproducing one NEDS decoding number on one session (Phase 0 task 1)

**What was tried:** Cloned NEDS (`external/NEDS`, gitignored) and ran its own
pipeline on held-out test session `d23a44ef-1402-4ed7-97f5-47e9a7a504d9`, on
this Mac: `prepare_data.py` → `create_dataset.py` → `train.py` for 1 epoch as a
smoke test, to measure cost before a full run.

**Hypothesis:** The NEDS paper reports a decoding number for an identifiable
held-out session, and running the released code on that session with default
settings would land close to it.

**Result:** The pipeline runs end to end, but **a numeric reproduction is not
possible as specified.**
- Single-session results are published **only as averages and distributions
  over the 10 held-out sessions**. Figure 2B's per-session scatter plot doesn't
  identify sessions.
- Those numbers come from a random hyperparameter search over 50 models.
- No linear or reduced-rank baseline code is released.
- Getting the pipeline to run at all took four distinct local patches (native
  build pin, `ibllib` API change, macOS `spawn` loop, empty Hugging Face org),
  all in `docs/DECISIONS.md`.
- A fresh ONE download serves data revisions that postdate the paper (motion
  energy from May–June 2025), so even a perfect code match wouldn't guarantee
  identical inputs.
- Full single-session training would take roughly 11–14 h on this machine's
  GPU (MPS).

**Why it failed (best guess):** The roadmap task assumed per-session published
numbers, and they don't exist. It isn't a flaw in NEDS's science.

**Do not retry unless:** per-session NEDS numbers with eids become available
(e.g. from the authors), or you have a CUDA GPU and budget for the 10-session,
50-model-search setup. Otherwise compare in Phase 3/4: run our baselines on
NEDS's 10 test eids (`external/NEDS/data/test_eids.txt`) and compare against
the paper's 10-session averages, using balanced accuracy explicitly (see
`docs/PRIOR_ART.md` §C).

---

## Template

### YYYY-MM-DD — Short title

**What was tried:**

**Hypothesis:**

**Result:**

**Why it failed (best guess):**

**Do not retry unless:**
