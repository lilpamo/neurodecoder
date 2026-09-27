# Negative Results

Failed experiments and dead ends. The most valuable and easiest-to-lose record
in this repo. One entry per experiment, newest first.

---

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
