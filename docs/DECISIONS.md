# Decisions

Record every non-obvious decision here: added dependencies, deviations from
CLAUDE.md, and choices about scope from §2. One entry per decision, newest
first.

---

### 2026-09-29 — The data provider, trial-structure decoders, and a contract fix (`neurodecoder/evaluation/data.py`)

**Decision:** `SplitData(split, target, context_bins=…)` is the real
`DataProvider`. It prepares every session of the split once:
- load it through the BWM backend;
- apply unit QC and bin it (`preprocess_session`);
- build the target (`targets/`);
- fit one `Normalizer` on the split's training data.

It keeps binned counts and targets, not spike trains, and normalises on each
request (about 150 MB per session instead of about 500 MB). Preparing
everything up front suits Phase 3's within-session split; Phase 4's
cross-session runs over hundreds of sessions will need a lazier provider.

- **Per-bin samples** (wheel velocity, movement state) are every window end in
  the partition's span whose bin lies in the task period and has a defined
  target. That's "task period, every bin" as the user chose.
  - `train_stride` (default 1) thins training samples only. How dense movement
    state's training should be is still the user's call.
- **Trial samples** (choice, block) are the target's usable trials that the
  split lists in the partition, at the target's own window. The context is
  forced to match the window: 5 bins for choice, 15 for block.
  - A trial whose window reaches outside the partition's span is dropped and
    counted (`dropped`), never silently. In the fixture, that happens when
    block's window starts before the train block does.
- **Shifts:**
  - shifted samples use the null's rotation (`evaluation.nulls`) at the same
    ends; a bin whose rotated target is undefined is dropped;
  - trial labels rotate over the session's usable trials before the partition
    filter;
  - `shifts()` draws within the task period (per-bin) or over the usable trials
    (trial targets).
- **`TrialStructureRidge` / `TrialStructureLogistic`** are the
  `null_trialstruct` decoders. They're the per-session baselines fitted on
  `task_features` only, with the same gapped CV; the contract hands them data
  with `z = None`.
- **Contract fix (from #22), found by the end-to-end test.** A shift draw that
  no session could take crashed the contract. That happens whenever every
  session is too short, e.g. choice with fewer than 200 usable trials. Such a
  draw is now undefined: NaN for every session, and the verdict line says
  "undefined, no test session has a defined null_shuffle" instead of claiming
  the model failed to beat it.

### 2026-09-28 — Baselines: ridge/logistic and multi-session RRR (`neurodecoder/models/baselines/`, `configs/baselines.yaml`)

**Decision (the user chose each option below before any code):**

| Choice | Setting |
|---|---|
| Per-bin inputs (wheel velocity, movement state) | each unit's summed activity in five 200 ms chunks of the last 1 s (context 50 bins) |
| Trial inputs (choice, block) | each unit's count over the target's window (the Brain Wide Map decoders' input); RRR sees the window bin by bin |
| Tuning | gapped blocked 5-fold CV inside the training data only: contiguous folds, with a max(context, 2 s) gap dropped either side of each validation fold; lowest mean validation MSE / log loss wins |
| Scope | ridge/logistic and RRR; the Poisson GLM is deferred (not a contract row, and it's an encoding model) |

- **Per-session models.** Ridge and logistic are fit per session: per-unit
  weights don't transfer across sessions. Predicting a session the model wasn't
  trained on raises, which is enough for Phase 3's within-session table. **Cross-session
  baselines (e.g. region-pooled features) are a Phase 4 decision.**
- **Penalty scale, corrected before any test metric existed.** The first grid
  put α on sklearn's summed-loss scale, from 10⁻³ to 10⁵. On `d23a44ef`'s wheel
  velocity, training-only CV picked 10⁵, the top edge; the CV optimum was 10⁶,
  and the edge cost 24% in validation MSE. Features are now standardised on
  the training data, and the penalty is per sample (mean loss + λ‖w‖²), with λ
  from 10⁻⁴ to 10⁴ in 17 steps. So one λ means the same for 230 trials or 78k
  bins. The same session now picks λ = 31.6, well inside.
- **Edge flag:** a λ at either end of the grid is flagged in the model
  (`at_grid_edge`), not silently kept.
- **Ridge** solves every λ and fold from one Gram matrix per session (19 s on
  `d23a44ef`'s 78,836 training bins with 1,950 features). It matches sklearn's
  `Ridge` on standardised features.
- **Logistic early stopping.** Logistic walks λ from strong to weak,
  warm-started, and stops a fold once its validation loss has risen twice in a
  row. The unreached weak end is where fits are slowest and never chosen; a λ
  not reached by every fold can't be selected. Movement state on `d23a44ef` costs:
  - 56 s using every 5th bin;
  - 297 s using every bin, with λ = 0.32 either way.

  **Open question for the first table:** with 20 shuffle refits of the model
  under test, training on every bin costs ~2 h per session for movement state.
- **RRR in two steps.** Session-specific unit weights U_s and shared temporal
  filters V (weights U_s Vᵀ over unit × chunk):
  1. per-session ridge on all features;
  2. V = top-r right singular vectors of the stacked session weights;
  3. U_s refit on X_s V (ridge, or logistic for classification) with the
     session's step-1 λ.

  The rank comes from {1, 2, 3} by the same folds (V re-estimated per fold), on
  the mean over sessions of relative validation MSE (or log loss). Alternating
  least squares with the same CV would cost hours on per-bin targets.
- **RRR measured:** 64 s per session on wheel velocity. On `d23a44ef` plus one
  other session it chose rank 1, with a filter weighted on the last 200–400 ms.
  Synthetic tests check it recovers planted rank-1 and rank-2 shared filters.

### 2026-09-28 — The evaluation contract (`neurodecoder/evaluation/contract.py`)

**Decision (the user chose each option below before any code):**
`evaluate(task, *, model, baseline_ridge, baseline_rrr, trialstruct, ceiling,
seed)` is the only entry point that scores a model. Every row is a required
argument, and it always fits and scores all six rows in CLAUDE.md §5 order.

| Row | What is fitted |
|---|---|
| `null_shuffle` | **the model under test**, refit once per shift draw on targets shifted within each session (`evaluation.nulls`); per-session median over draws |
| `null_trialstruct` | the trial-structure decoder, given each session's data **with spikes and units removed** (`z = None`) |
| `baseline_ridge`, `baseline_rrr` | the baselines, same split |
| `model` | the model under test |
| `ceiling_within` | the model on a within-session split of exactly the test sessions; for a within-session task it *is* the model row, and the report says so |

- **Fits receive the train partition only**, loaded one session at a time
  (`SessionData`, via a `DataProvider`). Predictions are scored per test
  session, and invalid predictions (wrong shape, NaN, a probability outside
  [0, 1]) raise.
- **The shuffle row re-trains the model under test**, so it measures what that
  model extracts from target autocorrelation alone.
  - It uses 20 draws by default. A costly model may use fewer, but never fewer
    than 5.
  - A session's k-th draw is the same shift for its train and test data, so a
    within-session model sees one consistent rotation.
  - Draws are seeded per session from the run seed.
  - A session without a valid shift sits the null out: it isn't used for
    training that draw, and its shuffle metric is NaN.
- **Verdicts:** for each of `null_trialstruct`, `null_shuffle` and
  `baseline_ridge`:
  - the test is a one-sided Wilcoxon signed-rank test over test sessions on the
    per-session difference in the primary metric, at α = 0.05. The line also
    gives the median difference and the win count;
  - "beats" needs p < 0.05 **and** a positive median difference;
  - fewer than 5 sessions can never reach α (p ≥ 2⁻ⁿ), so they never count as
    beating, and the line says why.

  A failure prints CLAUDE.md §5's consequence plainly: "this is not decoding",
  "no signal beyond the target's own autocorrelation", or "a more complex model
  is not justified".
- **Primary metric: AUROC for classification** (threshold-free, robust to
  choice's 26/74 imbalance) and R² for regression. Balanced accuracy is still
  reported, and the NEDS comparison uses it.
- **Per-bin targets are scored on every task-period bin** with a defined target
  and a full window in its span, the same span the shuffle rotates in.
- **The split guard runs first,** on the task's split and the ceiling's, at the
  task's context length.

**Still to come:** the real `DataProvider` (sessions → normalised counts,
windows, targets, null features) and the baselines. Until then the contract is
tested on synthetic sessions with stub decoders.

### 2026-09-28 — Null inputs (`neurodecoder/evaluation/nulls.py`, `configs/nulls.yaml`)

**Decision (the user chose each option below before any code):** the module
builds the inputs for the contract's two null rows; the baseline models fit
them. Nothing was fitted or scored to make these choices.

**`null_shuffle`:** the target is circularly shifted within its session, which
breaks the neural-behaviour alignment and keeps the target's autocorrelation.
- **20 shifts per session**, drawn with an explicit seed, distinct and uniform
  over [minimum, length − minimum].
- **Per-bin targets rotate within the task period** and are undefined (NaN)
  outside it. The minimum shift is **30 s**, about 20× the slowest
  decorrelation measured on three sessions. Autocorrelation falls below 0.1
  after:
  - 0.4–0.5 s for wheel velocity;
  - about 1 s for |velocity|;
  - 1.2–1.3 s for movement state, in two of the sessions. The third never
    dropped below 0.1 within 100 s, which is slow drift a circular shift keeps.
- **Trial-level labels rotate over the target's trials,** and each trial's
  window stays at its own time. The minimum shift is **100 trials**:
  - that is longer than any biased block (21–99 trials, median 47) and the
    90-trial opening block;
  - sessions with fewer than 200 usable trials (about 5%) get no shuffle null,
    reported as missing.

**`null_trialstruct`:** features from task variables only, never spikes.
- **Per-bin targets:**
  - one-hot time since each trial start, stimulus onset and go cue, in 0.1 s
    steps to 3 s, then one "later" feature (95 features in total);
  - the current trial's signed contrast and block prior.

  Behaviour-timed events (first movement, response, feedback) and the choice
  are never read, so beating this null means spikes add more than task timing.
- **Choice:** the current signed contrast and block prior, plus the previous 10
  trials' stimulus side, choice and reward.
- **Block:** the previous 10 trials' side, choice and reward only.
  - The block window ends before stimulus onset, so this null gets neither the
    current stimulus nor the label. The history is what reveals the block (80/20
    stimulus sides).
  - Stimulus side is taken from which contrast column is set, so a 0% trial
    still has a side, which follows the block prior like any other trial.
- **Movement onset** has no null yet; asking for one raises.

**Tests:**
- the features don't depend on spike counts, and don't change when
  behaviour-timed columns are permuted;
- history uses only earlier trials;
- the block null never sees its trial's own block, stimulus, choice or reward;
- shifts keep autocorrelation (lags 1–25) while decorrelating from the original.

**Smoke run on `d23a44ef`:** per-bin features are 183,447 × 95 (70 MB,
0.11 s). Every target gets 20 valid shifts: 115,658 task-period bins for the
per-bin targets, 290 trials for choice and 224 for block (block's valid shifts
are 100–124).

### 2026-09-28 — Decoding metrics (`neurodecoder/evaluation/metrics.py`, `configs/evaluation.yaml`)

**Decision:** metrics are computed **per session**, and summarised only as a
distribution across sessions: median, quartiles, range and the number of
sessions where the metric is defined. Nothing pools samples across sessions.
`test_no_pooled_r2_across_sessions` shows why: a decoder that only knows each
session's mean scores R² > 0.9 pooled, and ≈ 0 in every session.

- **Classification (binary):**
  - balanced accuracy, AUROC, F1, log loss and ECE, computed from
    p = P(class 1);
  - the hard prediction is class 1 when p ≥ 0.5, the plain argmax, so there's
    no threshold to tune.
- **ECE is top-label:** the predicted class's confidence, in 10 equal-width bins
  `[k/10, (k+1)/10)` (the last closed). 10 is netcal's default, which Phase 6
  will use; the bin count is `configs/evaluation.yaml: ece_bins`.
- **Regression:** R² (against the session's own mean), Pearson correlation, MAE
  and RMSE.
- **Undefined is NaN, never an error or a default:**
  - AUROC and balanced accuracy when a session has one class;
  - R² and correlation for a constant target;
  - correlation for a constant prediction;
  - F1 when there are no positives at all.

  Inputs with NaN are refused, so undefined samples (e.g. unlabelled
  `movement_state` bins) must be dropped explicitly first.
- **scikit-learn** (in the Fixed stack and `pyproject.toml`, but never installed
  in the local venv until now) supplies the standard metrics. Tests check each
  against it.

### 2026-09-28 — Targets (`neurodecoder/targets/`, `configs/targets.yaml`)

**Decision (the user chose each option below before any code):** five targets,
all built from the BWM release that the manifest and splits already use. The
release's wheel is within 0.52 mrad of raw, and its `firstMovement_times` is
within 1.7 ms of ONE's 2025-03-03 revision (see "BWM compressed backend").

| Target | Kind | Definition |
|---|---|---|
| `wheel_velocity` | per bin, rad/s | (position at bin end − position at bin start) / bin width; positions linearly interpolated at bin edges, never extrapolated (NaN outside the wheel's samples) |
| `movement_state` | per bin, 1 / 0 / NaN | 1 if the bin lies wholly inside a wheel movement, 0 if wholly inside a quiescent period, NaN otherwise; epochs from IBL's detector as shipped in `bwm_behavior` 2.0.0 (ibllib 4.0.1 `extract_wheel_moves`, quiescence ≥ 0.2 s) |
| `choice` | per trial, 1 / 0 | clockwise (`choice == +1`) / counter-clockwise (−1); window: the 100 ms before first movement |
| `block` | per trial, 1 / 0 | left block (`probabilityLeft == 0.8`) / right (0.2); the unbiased 0.5 block has no label; window: 0.4 s to 0.1 s before stimulus onset |
| `movement_onset` | per trial, event time | `firstMovement_times`, with the bin containing it; no window (how to score an event target is a Phase 3 choice) |

- **Wheel filter: bin displacement.** The alternatives were:
  - IBL/NEDS: interpolate to 1 kHz, 8th-order 20 Hz Butterworth run forwards and
    backwards, differentiate, and sample at the bin end. Each value then mixes
    in tens of ms of future movement, and IBL's filter settings come with it.
  - Causal smoothing over the last N bins: adds a lag and a new tunable.

  Bin displacement is the exact mean velocity over each spike bin, has nothing
  to tune and uses no sample from after the bin. It differs from NEDS's target
  (NEDS decodes |v| after the Butterworth filter), so the Phase 3 NEDS
  comparison must rebuild NEDS's target to compare like with like.
- **Trials: `bwm_include` only.** It is the BWM paper's rule (reaction time
  0.08–2 s, a choice made, no missing key events). It matches that rule on
  99.98% of the release's 295,920 trials (71 differ) and keeps 66.4%. NEDS's
  looser rule (reaction time 0–10 s, trial ≤ 10 s) would keep 83.9%.
- **Windows: the BWM paper's, in `configs/targets.yaml`.** The block window is
  pre-stimulus, so activity carrying the upcoming choice (which correlates with
  block) can't stand in for block. NEDS's window (0.5 s before to 1.5 s after
  stimulus onset, for every target) is left for the Phase 3 comparison, as an
  alternative config.
  - **Exact bin arithmetic:** a window is a whole number of bins (block 15,
    choice 5, at 20 ms). Its `end_bin` is the last bin that finishes by the
    window's stop, so it never reaches past it. It may start up to one bin
    before the window's start.
  - **Handing off to windows:** `end_bin` goes to `preprocess.windows`
    `extract` as `ends`, with `context_bins` from the target.
- **Movement state: moving / quiescent from IBL's epochs.** In all 459
  sessions, movements never overlap each other or a quiescent epoch, and
  quiescent epochs are exactly the gaps of at least 0.2 s between movements
  (plus the stretch before the first). A bin straddling an epoch edge is
  unlabelled, which adds no threshold.
- **Versioning:** `TARGETS_VERSION = 1`. `TargetConfig.fingerprint()` hashes it
  with the windows, and every target records it alongside the preprocessing
  fingerprint of the bins it sits on (R6).

**Checked on `d23a44ef`** (no decoding metric computed):
- **Wheel velocity** is defined in every one of the 183,447 bins. The smallest
  non-zero |v| is 0.003 rad/s, because positions are interpolated at bin edges
  3.1 ms off the wheel's 100 Hz samples.
- **Movement state** over the task period: moving 35.0%, quiescent 63.6%,
  unlabelled 1.4%.
- **IBL's detector and our velocity agree.** They are independent
  constructions from one wheel:
  - mean |v| is 1.079 rad/s in moving bins and 0.018 in quiescent ones (61×);
  - 99.3% of moving bins have non-zero velocity;
  - 65.5% of quiescent bins have exactly zero; the rest hold sub-threshold
    jitter that IBL's detector allows.
- **Trial targets:** 410 trials, 290 `bwm_include`. That gives 290 choice
  labels (26% clockwise), 224 block labels (46% left; the 66 unbiased-block
  trials have none) and 290 onsets.
- **Onsets agree with IBL's movement epochs:** 98.6% lie inside one (allowing
  one bin before its start), a median 2.0 ms from the nearest movement start.

### 2026-09-28 — Held-out configuration split, with within-lab percentiles (`neurodecoder/splits/registry.py`)

**Decision (definition from the user):** a session's configuration is its count
of QC-passing units under the task-period rule. `held_out_config(manifest,
units, preproc)` works from `release_units` without loading sessions:
- **Test:** sessions below the 10th percentile **of their own lab's** sessions.
- **Train:** sessions at or above their lab's 25th percentile.
- **Dropped:** sessions in between.

Animals may be shared between train and test, as for `held_out_region`. The
split records each lab's cutoffs: its percentiles, the same cutoffs as whole
unit counts, and its session count. It also records every session's count and
lab. The guard recomputes every lab's cutoffs from those records, and checks
each session against its own lab's.

**Why within lab (option chosen by the user):** percentiles taken across the
whole release gave 46 test sessions (≤ 46 units) and 344 train (≥ 90). But the
test set was 30.4% hausserlab (13.5% of sessions) and 17.4% wittenlab (8.1%),
so the two labs made up 48% of it. The cause is lab protocol, not yield:
- 95% of hausserlab's sessions use one probe, so it has the lowest session
  counts (median 91.5, against 128–234 elsewhere), yet its units per probe (83)
  are typical;
- 43 of those 46 test sessions were single-probe.

So that split would have measured lab shift. Units per probe still left two
labs at 37% of test. Within-lab percentiles bring every lab's test share to
within 1.3 points of its share of the data.

**On the full manifest** (459 sessions; built in 0.4 s; passes the guard):
- **Sizes:** 50 test, 346 train, 63 dropped.
- **Test animals:** 40. 31 have one test session, 8 have two and 1 has three.
  38 of the 50 test sessions come from animals with other sessions in train.
- **Unit counts:** test median 32 (2–109), train median 165.5.
- **Probes:** 90% of test sessions are single-probe, against 38% of train and
  48% overall. The split still mostly holds out single-probe recordings, but
  now within each lab, which is the configuration shift it's meant to measure.

| Lab | Sessions | Test below (≤ units) | Train from (≥ units) | Test | Train | Test share | Share of all |
|---|---|---|---|---|---|---|---|
| angelakilab | 41 | 61.0 (60) | 90.0 (90) | 4 | 31 | 8.0% | 8.9% |
| churchlandlab | 36 | 71.0 (70) | 112.0 (112) | 4 | 27 | 8.0% | 7.8% |
| churchlandlab_ucla | 41 | 77.0 (76) | 122.0 (122) | 4 | 31 | 8.0% | 8.9% |
| cortexlab | 42 | 88.1 (88) | 97.0 (97) | 5 | 32 | 10.0% | 9.2% |
| danlab | 46 | 63.0 (62) | 90.0 (90) | 5 | 34 | 10.0% | 10.0% |
| hausserlab | 62 | 32.4 (32) | 55.75 (56) | 7 | 46 | 14.0% | 13.5% |
| hoferlab | 17 | 46.6 (46) | 90.0 (90) | 2 | 13 | 4.0% | 3.7% |
| mainenlab | 44 | 65.4 (65) | 122.75 (123) | 5 | 33 | 10.0% | 9.6% |
| mrsicflogellab | 25 | 44.8 (44) | 73.0 (73) | 3 | 20 | 6.0% | 5.4% |
| steinmetzlab | 31 | 17.0 (16) | 70.0 (70) | 3 | 23 | 6.0% | 6.8% |
| wittenlab | 37 | 31.4 (31) | 79.0 (79) | 4 | 28 | 8.0% | 8.1% |
| zadorlab | 37 | 110.2 (110) | 134.0 (134) | 4 | 28 | 8.0% | 8.1% |

**Consequence:** "few units" is relative to the lab. A zadorlab test session
(≤ 110 units) can hold more units than a hausserlab training session (≥ 56).
The split measures low yield for the lab's protocol, not an absolute unit
count.

### 2026-09-28 — Unit QC uses the task-period firing rate (`PREPROC_VERSION` 2)

**Decision (signed off by the user on the evidence below):** unit QC's 0.1 Hz
floor now applies to each unit's firing rate during the task: spikes from the
first trial's `intervals_0` to the last trial's `intervals_1`, both included,
over that time. `min_label` (1.0) and `exclude_regions` (void, root) are
unchanged. The release's `firing_rate` (spike count over the whole recording,
first spike to last) stays in the units table as a reported column only.
`PREPROC_VERSION` goes to 2, so every earlier fingerprint, and any split built
against one, is refused.

**Why:** the normalisation work found units that pass the old floor but are
nearly silent during trials. Recordings run before and after the task (the task
is a median 70% of the recording, 28–111 min), and some units fire almost only
then.

**Evidence** (all 75,395 good units in `bwm_ephys` 1.2.1; measured in 63 s):

| | Whole-recording rule (v1) | Task-period rule (v2) |
|---|---|---|
| Units removed | 83 | 1,402 |
| Pass both / only v1 / only v2 / neither | 73,964 / 1,348 / 29 / 54 | |
| Per-session loss, 50th / 90th / 95th / max (share of good units) | | 1.0% / 4.4% / 6.1% / 34.6% |
| Sessions losing 0 / < 1% / 1–5% / 5–10% / 10–25% / > 25% | | 154 / 231 / 193 / 24 / 10 / 1 |
| QC-passing units per session, 10th / 25th / 50th / 75th / 90th | 48 / 91 / 141 / 217.5 / 304.4 | 46.8 / 89.5 / 138 / 213 / 296.2 |

- A typical unit's two rates agree: the median task/whole ratio is 1.00
  (10th–90th percentile 0.63–1.33).
- **15 sessions lose more than 25% or end with fewer than 20 units.** 14 of
  them already had fewer than 20 under v1, and v2 removes at most one unit from
  each.
- **The outlier is `0cc486c3` (ibl_witten_29, wittenlab), 131 → 87 units.** Its
  46 lost units fire at a median 0.72 Hz over the recording, but only 0.1% of
  their spikes fall inside the task (53–2,870 s of a 0–4,400 s recording).
- **On `d23a44ef`, 397 → 390 units, identical across the BWM, NWB and ONE
  backends.** `probe00_27` (0.0965 Hz whole-recording, 0.0013 Hz in the task)
  was already out. The new rule removes 6 of the 9 near-silent units from
  "Normalisation" plus `probe01_280`. (The chat report said 7 of 9; it is 6.)

**Where the rates come from:**
- **Loaded sessions:** `apply_unit_qc` computes the rates from the session's own
  spikes and trials, and keeps them as a `task_firing_rate` column. No backend
  or session cache changes.
- **Release-wide builders** (`held_out_region`, `held_out_config`) must not load
  sessions, so they read a precomputed table (location chosen by the user):
  - it lives at `<data_root>/derived/bwm_ephys-1.2.1/task_rates-v1.parquet`
    (1.3 MB), with `task_rates-v1.provenance.json` beside it;
  - it's built once by `python -m neurodecoder.cli.build_task_rates` (54 s,
    6 processes, decoding every spike shard);
  - `load_task_rates` refuses a table whose provenance (release name and
    version, `TASK_RATES_VERSION`, unit count) doesn't match;
  - `release_units` joins it onto `units.parquet` and refuses mismatched units.
  After the join, a split build takes about as long as before (0.06 s).
- **Both paths use the same definition.** Both call `qc.units.task_period` and
  `in_task`, and the table uses the BWM backend's shard decoder. On `d23a44ef`
  the table and the session path agree exactly for all 398 units (tested).

**Caches and splits:** nothing on disk depended on preprocessing. The session
cache is keyed on backend, source and loader version, and holds pre-QC
sessions, so it stays valid. No split files had been saved.

**`held_out_region`, rebuilt for all 266 regions:**
- **The four regions with ≥ 20 test sessions are unchanged:** CP 43 / 358 (and
  still 30 parent-label sessions), MRN 33, APN 26, PO 23.
- **13 regions change test size by one or two sessions.** Examples: CA1 14 → 15,
  PRNr 13 → 12, PRM 5 → 3. PRP and SPIV lose their only test session; PC5 gains
  one.
- **Regions with a test set:** 144 → 143. With ≥ 10 test sessions: 16 → 16.
  With ≥ 5: 42 → 41.
- **Training sets:** among the 142 regions with a test set under both rules, 22
  gain one training session and one loses one.

**Tests whose hardcoded counts this rule changes:**
- `test_unit_qc::test_real_bwm_floor…`: 83 → 1,402 removed.
- `test_unit_qc::test_real_three_backends…`: 397 → 390.
- `test_binning::test_real_round_trip…`: 397 → 390.

The CP counts in `test_region_split` don't change. Fixtures changed only to
give QC a `task_firing_rate` or a real task period (`test_unit_qc`,
`test_binning`, `test_region_split`).

**Known and not acted on: within-session drift.** Units that appear or vanish
partway through the task still pass. `probe00_514` on `d23a44ef` fires at 0.011
Hz in the `within_session` train block and 2.07 Hz in the test block (task rate
0.66 Hz), giving z-scores up to 179. 57 of its session's units have a test-block
mean z beyond ±0.5. A per-block rate criterion would tie QC to a split, so drift
is logged here rather than filtered.

### 2026-09-28 — Windows (`neurodecoder/preprocess/windows.py`)

**Decision:** `window_plan(split, context_bins=…, stride_bins=…)` runs
`assert_split_valid` once for that context and returns a plan. For a session
and partition, the plan allows a span of bins:
- **`within_session`:** the partition's block.
- **Other kinds:** the whole binned session, and only if the session is in that
  partition.

`extract(z, binned, partition, ends=None)` returns `(n_windows, n_units,
context_bins)` float32 windows. It refuses any end whose window would leave the
span. `unwindow` puts windows back on the session's bins.

- **A window ending at bin e covers bins e − context + 1 … e.** Default ends
  are every `stride_bins`-th bin from the first full window. Where targets sit
  relative to a window, and which ends have a target, is left to `targets/`.
  `extract` takes ends chosen elsewhere and still enforces the span.
- **Context and stride are required,** with no defaults. Their values belong
  in the run configs that come with models; the context also feeds the guard.
- **Windows come from a strided view,** and indexing copies only the chosen
  windows. **Batching is the caller's job:** on `d23a44ef`'s train block (78,885
  bins, 397 units) windows take 0.13 GB at stride 50, 1.25 GB at stride 5, and
  would take about 6 GB at stride 1. The data loader should pass `ends` in
  batches.
- **For cross-session splits the span is the whole binned session,** including
  time before the first trial and after the last. Restricting to the task
  period depends on the target, so `targets/` does it by choosing ends.

**Verified on `d23a44ef`:**
- The Phase 2 criterion "round-trip a session to tensors and back to spike
  counts within float tolerance" passes: counts → normalised → windows
  (context 50, stride 50) → `unwindow` → inverse → counts matches exactly on
  every bin a window covers.
- Train and test windows share no bin and are at least a context apart (the
  roadmap's `test_no_temporal_leakage`, now at the window level).
- Train gives 1,577 windows and test 731, extracted in ≤ 0.1 s.

### 2026-09-28 — Normalisation (`neurodecoder/preprocess/normalize.py`)

**Decision (option chosen by the user):** per-unit statistics where the units
have training data, pooled statistics otherwise. `fit_normalizer(split, binned,
preproc)` takes a split, never a bare session, and reads only training data: the
train block of each `within_session` session, or the sessions of the train
partition. Calibration and test data are never read.

The mode is set **per split kind**:
- **`within_session` → `per_unit`:** each unit's mean and std over its train
  block.
- **Every cross-session kind → `pooled`:** one mean and std over all units and
  bins of the train partition, applied to every unit, train and test alike.

The per-split-kind mode is how the user's chosen option reads ("the
within-session ceiling is normalised more finely than cross-session rows").
Its reason: normalising training units per unit and unseen test units pooled
would itself shift the inputs between train and test.

- **std floor = √(min_firing_rate_hz · bin_s)**, the std of a Poisson unit at
  the QC minimum rate (0.045 at 0.1 Hz, 20 ms). It's derived from existing
  config, so there's no new tunable. Without it a unit silent in training would
  divide by zero.
- **Sums are exact int64 sums,** so the statistics don't depend on session
  order. The `Normalizer` is a hashed dataclass that serialises to JSON for the
  model artifact (R3). `from_dict` refuses edited content.
- **`transform` returns `(n_units, n_bins)` float32** (291 MB for
  `d23a44ef`). It refuses binned data with another fingerprint, and in
  `per_unit` mode a session or unit set it wasn't fit on. `inverse_transform`
  undoes it.

**Verified on `d23a44ef`** (`within_session`, 0.8, 2 s):
- Counts round-trip through `transform` and `inverse_transform` exactly after
  rounding. This is half of Phase 2's success criterion "round-trip a session
  to tensors and back"; the windows module is the other half.
- Fit takes 0.04 s and transform 0.22 s.

**Finding, not acted on: within-session non-stationarity.**
- **9 of 397 units hit the std floor.** They are nearly silent in the train
  block (≤ 0.07 Hz) but pass QC's 0.1 Hz. QC uses the release's whole-recording
  firing rate, which includes the post-task period.
- **Some fire mostly after the task.** `probe01_953` spikes only from 2,637 s,
  and the trials end at 2,330 s. `probe01_1157` averages 9 Hz overall but
  0.01–0.18 Hz during trials.
- **Some appear partway through,** probably from drift. `probe00_514` is at
  0.011 Hz in train and 2.07 Hz in test, giving z-scores up to 179.
- **57 units have a test-block mean z beyond ±0.5.**

A task-period firing-rate criterion would change the signed-off QC, so it's
left for the user.

### 2026-09-28 — Held-out region split, and `iblatlas` as a dependency (`neurodecoder/splits/registry.py`)

**Decision:** `held_out_region(manifest, units, preproc, region=R)` follows the
split doc's stricter definition, with one tightening. `units` is the BWM
release's `metadata/units.parquet`, the same table the BWM backend loads
sessions from. Only units that pass `preproc.qc` count.
- **Test:** at least 20% of the session's QC-passing units have Beryl region R.
  The denominator includes units with no Beryl region (fibre tracts, ventricles,
  parent-only labels).
- **Train:** no QC-passing unit in R, **and none that could be in R.** A unit
  could be in R when it has no Beryl region and its Allen label contains R or
  lies inside it: `STR` for CP, `TH` for PO, `MB` for MRN.
- **Everything else is dropped.** The split still records every session's
  counts (`n_units`, `n_in_region`, `n_possibly_in_region`), so the file shows
  why a session is out. The guard re-checks both rules from those counts.
- **R must be a Beryl region.** The split records R, the 20% threshold and the
  `iblatlas` version.
- **No calibration partition,** as for `held_out_session`.

**Why the tightening:** 10,059 of the 75,395 release units (13%) have no Beryl
region. Most are fibre tracts, but many carry only a parent label: `MB` 1,465,
`P` 751, `STR` 526, `TH` 310. Under the literal "zero units in R" rule such a
session trains while possibly holding units from R. Excluding them costs
training sessions:
- 30 for CP (388 → 358);
- 50 for MRN (349 → 299);
- 107 for APN (408 → 301);
- 42 for PO (409 → 367);
- none for cortical regions such as MOp, MOs and VISp.

**Why `iblatlas`:** the containment test needs the Allen hierarchy. `iblatlas`
is IBL's atlas package, a dependency of `ibllib` (already in the Fixed stack),
though `ibllib` itself isn't installed here. 1.2.1 is the version already used
in NEDS's environment. It adds only `pynrrd` beyond the Fixed stack, plus
matplotlib (Fixed, never installed until now). Its Beryl mapping agrees with
the release's `beryl_acronym` for all 75,395 units.

**Verified on the full manifest** (every Beryl region, 0.4 s each, all pass the
guard): 144 of 266 regions have a test set.
- **≥ 20 test sessions: 4 regions.** CP 43 (358 train), MRN 33, APN 26, PO 23.
- **≥ 10: 16 regions.**
- **≥ 5: 42 regions.**

**Test animals are not kept out of training.** For CP, 36 of the 43 test
sessions come from animals with other sessions in train. The split doc doesn't
ask for animal disjointness. Without it, a held-out-region result measures
region transfer to animals the model has already seen, so it can't be read as
cross-animal. Requiring disjointness would shrink test sets further.

**Signed off by the user (2026-09-28):** both the parent-label tightening and
leaving animals shared between train and test, as described above.

### 2026-09-28 — Split registry and guards (`neurodecoder/splits/`)

**Decision:** four of the six split kinds in `docs/SPLITS_AND_LEAKAGE.md` are
built by `splits/registry.py`, checked by `splits/guards.assert_split_valid`,
and saved as JSON with a hash.

| Kind | Builder | Partitions |
|---|---|---|
| `held_out_animal` | `held_out_groups(by="subject", n_test, n_calibration, seed)` | whole animals; calibration is its own set of animals |
| `held_out_lab` | `held_out_groups(by="lab", …)` | whole labs, the same way |
| `held_out_session` | `held_out_session(seed)` | one session of each animal with ≥ 2 sessions is test; everything else is train |
| `within_session` | `within_session(trials, train_fraction, gap_s)` | per session, one early train block and one late test block |

- **Seeds are required keyword arguments** with no default (R7). Groups are
  sorted and then shuffled with `numpy.random.default_rng(seed)`. The saved
  file, not the seed, is the record: another numpy version could draw
  differently from the same seed.
- **Group splits cover the whole manifest** (all 459 sessions). Filtering for
  a task, such as sessions with pose, happens downstream and must not re-split.
- **`within_session` blocks are bounded by trials.** The train block runs from
  the first trial's start to the end of the last train trial
  (`round(train_fraction · n_trials)` trials). The test block starts at the
  first later trial whose first bin is at least `ceil(gap_s · rate)` empty bins
  after the train block. Trials in between belong to neither partition, and so
  does time before the first trial or after the last. `gap_s ≥ 2 s` is
  enforced when the split is built.
- **No calibration partition yet for `held_out_session` or `within_session`.**
  Phase 7's cross-animal conformal needs one only for animal splits. It can be
  added when a within-session calibration result is actually needed.
- **The hash** is the sha256 of canonical JSON (sorted keys, no NaN) of the
  whole split: kind, params, partitions, per-session records, the manifest
  provenance and the preprocessing `{fingerprint, bin_ms}`. The fingerprint
  covers `PREPROC_VERSION`. `load_split` refuses a file whose content no longer
  matches its hash. **`save_split` never replaces an existing file with a
  different split**, so a split used by a result can't change under it.
- **The guard raises and never warns.** It checks:
  - the split's `manifest_version`, release versions and preprocessing
    fingerprint equal the current ones (by default, this code's and
    `configs/*.yaml`'s);
  - partitions are known, non-empty and include train and test;
  - for group splits, no subject or lab is shared by any pair of partitions,
    calibration included;
  - for non-temporal splits, no session is shared;
  - for `within_session`, blocks share no bin and have at least
    `max(context_bins, 2 s)` empty bins between them;
  - each listed trial is in exactly one partition and lies wholly inside that
    partition's block.

  Each check has a test that feeds it a broken split.
- **The split file stores every trial's interval,** so the guard can check
  trial integrity without loading sessions. That makes a `within_session`
  file over all 459 sessions 12.6 MB. A `held_out_animal` file is 0.07 MB.

**Verified on the full manifest (459 sessions, 139 animals, 12 labs):**
- **`held_out_animal`** (20 test, 15 calibration, seed 0): 354 / 52 / 53
  sessions. The roadmap asks for ≥ 20 test sessions.
- **`held_out_lab`** (2 test, 1 calibration): 343 / 44 / 72 sessions.
- **`held_out_session`**: 352 train, 107 test.
- **`within_session`** on every session (0.8, 2 s): at most 1 trial per
  session falls in the gap. Actual gaps range from 3.4 s to 66.3 s (median
  4.8 s), because the test block starts at a trial start.
- Every split passes the guard at a 50-bin context. Every split builds in under
  a second, from a manifest that builds in 0.7 s.

**Not built yet:**
- **`held_out_region`** needs per-session counts of QC-passing units per
  region. The rule is ≥ 20% of units in R for test and zero units in R for
  train. The manifest's `regions` come from the release's good units, not from
  our QC.
- **`held_out_config`** needs a definition of a "standard" probe configuration.

The roadmap's Phase 2 success criterion ("all five split types generate and
validate") isn't met until both exist. The split doc's table lists six kinds,
so the roadmap's "five" is out of date by one.

### 2026-09-28 — Binning and `PREPROC_VERSION` (`neurodecoder/preprocess/binning.py`)

**Decision:** `preprocess_session(session, config)` applies unit QC and then
bins spikes into `(n_units, n_bins)` `uint16` counts. `bin_spikes` bins
without QC, and its result carries no fingerprint.

- **The grid is anchored at t = 0 on the session clock:** bin *k* covers
  `[k·w, (k+1)·w)`, and `first_bin = floor(t_start · rate)`. Backends whose
  time bounds differ slightly still line up bin by bin (BWM and NWB differ by
  one bin at the end on `d23a44ef`). A spike exactly on an edge goes to the
  upper bin.
- **The bin width is an integer number of ms that divides 1000**
  (`configs/preprocess.yaml`, default **20 ms**, matching NEDS; 10 and 50
  also valid). A spike's bin is `floor(t · rate)` with an integer rate, which
  avoids `floor(t / w)` misplacing spikes through float error
  (`0.06 / 0.02 = 2.999…`).
- **Counts are `uint16`,** and a bin over 65,535 raises instead of wrapping.
  The real maximum on `d23a44ef` is 19 per 20 ms bin.
- **`PREPROC_VERSION = 1`** is a code version, bumped by hand whenever
  binning (or anything it calls) gives different output for the same input.
  **`PreprocConfig.fingerprint()`** is the sha256 of `PREPROC_VERSION`, the
  bin width and the unit-QC hash. It's what caches and the split registry
  will record (R6): changing any threshold, the bin width or the code version
  changes it.

**Verified on `d23a44ef` (20 ms):**
- `(397, 183,448)` counts, 146 MB, QC plus binning in 0.43 s.
- Each unit's counts sum to its spike count.
- **ONE and NWB bin byte-identically** (same spike times).
- **BWM differs from NWB in 0.248% of spikes** (64,293 of 25,887,688), each
  moved by one bin. BWM stores 100 µs ticks, so only spikes within 50 µs of
  an edge can move; the expected fraction for evenly spread rounding is
  50 µs / 20 ms = 0.25%. The test enforces the 0.5% worst case.
- The same input gives byte-identical counts and the same fingerprint (the
  roadmap's `test_binning_determinism`).

### 2026-09-28 — Phase 2 order, and unit QC (`neurodecoder/qc/units.py`, `configs/qc.yaml`)

**Signed off by the user before any Phase 2 code (CLAUDE.md §10):**
- **Module order:** `qc/units` → `preprocess/binning` (defines
  `PREPROC_VERSION`) → `splits/registry` + `guards` → `preprocess/normalize`
  → `preprocess/windows` → `targets/`. Binning comes before splits because
  the split file's hash covers `PREPROC_VERSION`, and splits come before
  normalisation because `normalize.py` takes a split object
  (`docs/SPLITS_AND_LEAKAGE.md`).
- **Unit QC**, with the thresholds fixed now, before any metric exists, so
  they can't be tuned against results (R6):

| Criterion | Value | Evidence at sign-off (BWM release) |
|---|---|---|
| IBL QC label | `== 1.0` | label = fraction of 3 metrics passed; **100%** of label-1 units pass IBL's sliding-RP test |
| Location | not `void`/`root` | 307 label-1 units are located there (see "Session manifest") |
| Firing rate | `≥ 0.1 Hz` | removes 83 of 75,395 good units (0.11%); 1 Hz would remove 8.3%, NEDS's 5 Hz 45.8% |
| Separate RPV ceiling | none | redundant with label 1 for IBL data; revisit for non-IBL NWB (Phase 11), which has no IBL label |

0.1 Hz was chosen to stay closest to BWM's published good-unit set, which
Phase 3's baselines will be compared against.

**Implementation:**
- `unit_qc(units, qc)` returns `passed` plus a `reason` naming **every**
  failed criterion. A missing label, firing rate or location value **fails**
  the unit instead of passing it (§7).
- `apply_unit_qc(session, qc)` keeps the passing units and their spikes, and
  raises if none pass (§10).
- **Location comes from whichever field the backend provides:** BWM's
  `acronym`, ONE's Allen `atlas_id` (0 = void, 997 = root) or NWB's full
  names in `location`. With none of them, QC raises rather than silently
  skipping the in-brain criterion.
- `UnitQC.hash()` changes whenever any threshold does. `PREPROC_VERSION`
  (next module) will incorporate it.

**Verified on `d23a44ef`:**
- After QC, **all three backends keep exactly the same 397 units**: BWM's 398
  good units minus `probe00_27` (0.0965 Hz).
- ONE's `atlas_id == 997` and NWB's `location == "root"` pick out the same
  units.
- Across the whole BWM release, QC removes exactly the 83 sub-0.1 Hz units.

**Known cross-backend residual:** BWM's good-unit table also drops 6 label-1,
in-brain units for an unrecorded reason (see "Session manifest"). QC on
ONE/NWB would keep those 6, so the unit sets can differ by them on the
sessions where they occur. `d23a44ef` has none.

### 2026-09-28 — BWM backend loads behaviour from `bwm_behavior` 2.0.0

**Decision:** `load_session_bwm(eid, root, behaviour_root=None)` decodes the
per-session `sessions/<eid>.zip` shard of the `bwm_behavior` 2.0.0 release
(version pinned through its `manifest.json`), following ibl-ai-agent's own
decoder. Unknown encoding kinds raise. `load_session("bwm")` passes the release
from config. `LOADER_VERSION` is now 2, and the cache source gains
`behaviour_version`, so every previously cached BWM session (including the 50
from the Phase 1 check) becomes a miss; those entries are orphaned on disk.

| Key | From the shard |
|---|---|
| `wheel` | `wheel.position` only, **not** the stored velocity, which bakes in IBL's smoothing filter (R6) |
| `pose_{left,right,body}` | each keypoint's `_x`, `_y`, `_likelihood` columns, under IBL's ALF names |
| `motion_energy_{left,right}` / `motion_energy_body` | `whiskerMotionEnergy` / `bodyMotionEnergy` |
| `pupil_left` | `pupilDiameter_raw`, **not** `pupilDiameter_smooth` (R6) |
| `pupil_right` | declared missing: the right camera has no pupil diameter in this release |
| `lick` | declared missing: not in `bwm_behavior` |

A source the build listed in a camera's `skipped_sources`, or a camera or
shard that's absent, is declared missing with that reason.

**What the release actually stores, measured against NWB on `d23a44ef`
(ibl-ai-agent's docs understate it):**
- **The wheel is resampled, not native.** It's the raw encoder position
  linearly interpolated onto an exact 100 Hz grid (366,885 samples versus
  NWB's 755,552 raw ones; only 15.7% of its times coincide with a raw sample)
  and rounded to 0.001 rad. It's within **0.52 mrad** of NWB's raw wheel
  interpolated at the same times (one encoder tick is 1.53 mrad).
  **Phase 2 consequence:** a wheel-velocity target differs slightly by
  backend. Build it from one declared source.
- **Camera times are an ideal grid, not the frame times.** Each camera keeps
  the real frame nearest each point of a uniform 60 Hz (left, right) or 30 Hz
  (body) grid, and stores `start + i/rate` as its time. The right camera is
  downsampled from ~150 Hz. Each grid time is within half a frame of its real
  frame's time (max 8.3 / 3.3 / 16.6 ms for left / right / body). If a dropped
  frame ever made two grid points pick the same frame, deduplication would
  shift every later time by a whole frame. It doesn't happen in this session
  (frame counts equal grid sizes), but the loader can't detect it from the
  shard alone.
- **Quantisation:** pose x/y to 0.5 px, likelihood to 8 bits, motion energy
  and pupil diameter to 0.05.
- **Right and body pose match NWB within quantisation** (x/y ≤ 0.25 px,
  likelihood ≤ 0.002) at the matching frames: same tracker output, correct
  decoding.
- **Left-camera pose does not match NWB, unexplained.** At the correct frames,
  x/y differ systematically: paw median ~11 px (max 447 px), pupil ~0.3–1.2 px
  (max 3.7 px). Likelihoods are mostly identical. It isn't a frame offset (the
  error is smallest at zero shift) and it isn't mirrored names (swapping
  left/right makes it much worse). The likeliest cause is different
  post-processing of left-camera x/y in one source, but it isn't recorded in
  either dataset. Don't mix left-camera pose across backends.
- **Pose NaNs differ by design.** NWB's converter blanks x/y where tracker
  likelihood is below **0.9** (NaN ⇔ likelihood < 0.9, exactly: max 0.898
  where NaN, min 0.902 where finite). BWM keeps the tracker's raw
  low-confidence estimates, as its docs say. Neither loader applies a
  threshold. To compare or combine backends, apply the same likelihood mask
  to both; choosing it is a Phase 2 preprocessing decision.
- **Keypoint names differ.** BWM uses IBL's ALF names (`paw_l`, `paw_r`,
  `pupil_{top,bottom,left,right}_r`, `tongue_end_l/r`, `nose_tip`,
  `tube_top`, `tube_bottom`, `tail_start`), while NWB's converter uses
  `left_paw`, `right_paw`, `right_pupil_*`, `left/right_tongue_end`, and the
  same names for the rest. BWM also has more left-camera keypoints (11 versus
  NWB's 6). Aligning NWB to the ALF names is a separate NWB backend change.

**Verified:** 11 tests.
- 6 run in CI on a synthetic shard in the release's exact format: every
  encoding kind, rejection of unknown kinds, canonical keys, position-only
  wheel, raw pupil, and skipped sources declared missing.
- 5 run on the real release against NWB: the wheel residual ≤ half a
  quantisation step; right and body pose within quantisation, with frame
  times within half a frame; left-camera frame timing within half a frame;
  and the capability report.

### 2026-09-28 — ONE backend (`neurodecoder/data/backends/one_backend.py`), and three-way agreement

**Decision:** `load_session_one(eid, one)` reads a session from IBL's public
Alyx through ONE, and `load_session(eid, "one")` registers it. **Revisions are
pinned** and both are part of the cache key:

| Pin | Value | Why |
|---|---|---|
| `SORTER_REVISION` | `2024-05-06` | the sorting NWB and the compressed BWM both use |
| `TRIALS_REVISION` | `2025-03-03` | the only trials revision on Alyx for these sessions, and the one NWB matches |

This is the direct answer to the Phase 0 finding that **ONE serves the newest
revision by default**, which is how NEDS's motion-energy inputs came from files
postdating its paper (`docs/PRIOR_ART.md` §C). Every `load_object` call here
passes an explicit revision; changing a pin is a deliberate edit that
invalidates cached sessions.

- **No new dependency.** It uses ONE-api only (already in the Fixed stack),
  not `ibllib`/`brainbox`. Phase 0 showed `ibllib`'s API drifts, and the
  high-level loaders aren't needed for this.
- **All units, not just good ones.** `units.label` carries IBL's QC label
  (1.0 = good), and callers filter. Extra columns: `cluster_id`,
  `cluster_uuid` (matching NWB's), `atlas_id` and `peak_channel`.
- **`units.x/y/z` are metres relative to bregma**, from each cluster's peak
  channel in `channels.mlapdv` (µm), the same convention as BWM. So ONE and
  BWM share a coordinate frame, while NWB's are Allen CCF µm.
- **`units.acronym` is declared missing:** ONE gives Allen CCF ids (kept as
  `atlas_id`), and mapping ids to acronyms needs `iblatlas`.
- **`spikes.clusters` indexes the cluster table, it is not `cluster_id`.**
  Getting that wrong would silently mis-assign every spike, so the backend
  range-checks the index and raises.
- **Behaviour:** the raw wheel only. A session with no wheel is declared
  missing rather than raising. Pose, pupil and motion energy aren't loaded
  yet.
- **`make_one(cache_dir)`** builds the client, caching downloads under
  `one_cache` from `configs/data.yaml` (`~/data/neurodecoder/one`).

**Three-way agreement on `d23a44ef`, now tested in code:**
- **ONE vs NWB: exact.** All 1,961 units, all 61,981,600 spike times
  element-for-element, all 13 trial fields, every `cluster_uuid` and `label`,
  the `depths`, and the 755,552-sample wheel. Both derive from the same
  sorting run, so exact equality is the right bar, and the NWB conversion
  preserves the data.
- **BWM vs NWB:** same good units and spike counts, spike times within 50 µs
  (BWM stores 100 µs ticks), all trial fields identical except
  `firstMovement_times` (see "BWM compressed backend").

This satisfies ROADMAP Phase 1's "three backends return byte-identical unit
counts and spike-count totals for the same eid", with the two documented,
explained discrepancies rather than a silent averaging over them.

**Cost:** about 60 s for this session from a warm ONE cache; the session's
files are about 1.6 GB, dominated by `spikes.times`/`spikes.clusters` for
both probes. The cache for the real tests was populated by copying the files
the Phase 0 NEDS run had already downloaded, rather than fetching them again.
Those tests skip when that cache is absent.

### 2026-09-28 — Phase 1 success check (`neurodecoder/cli/phase1_check.py`): passed

**Decision:** ROADMAP Phase 1's success criterion ("`load_session` works for
50 sessions across ≥10 subjects and ≥3 labs, from cache, in under 5 s each")
is checked by `python -m neurodecoder.cli.phase1_check`:
1. It builds the manifest and picks 50 sessions deterministically, with no
   randomness (R7): each subject's earliest session, taking labs in turn.
2. It loads each through `load_session(eid, "bwm")`, which fills the cache,
   then times a second, cached read.
3. It checks each session's unit and trial counts against the manifest.
4. It writes `runs/<UTC time>_phase1_check/report.json` with the git SHA and
   a dirty flag (§7), and exits non-zero on any failure.

The manifest needs the `bwm_behavior` folder, so `configs/data.yaml` gained
a `bwm_behavior` key, with `DataConfig.bwm_behavior_root` in `load.py`.

**Result, run 2026-09-28 (`runs/20260928T071223Z_phase1_check/`): PASSED.**
- 50 sessions from **50 subjects across all 12 labs**.
- **Cached reads: median 0.06 s, p95 0.25 s, max 0.37 s**, against the 5 s
  budget.
- First loads from source, including the cache write: median 0.50 s, max
  5.76 s (the largest session, 421 units).
- **All 50 sessions' unit and trial counts match the manifest.** Sessions
  range from 12 to 456 good units (median 145) and 402 to 1,440 trials.
- Cached read time rises with session size (correlation 0.81 with unit
  count), as real I/O should.
- Whole run 48 s; cache 3.1 GB for the 50 sessions.

**Caveats:**
- The cached reads came right after the writes, so the OS file cache was
  likely warm; clearing it needs admin rights. Cold reads will be slower,
  but the slowest warm read has a 13× margin.
- This run recorded `git_dirty: true`, because the check module wasn't
  committed yet (base commit `2ef7e03`). Rerun after committing for a report
  tied to committed code.

**What Phase 1 still lacks, even though this criterion is met:** the ONE
backend, behaviour in the BWM backend, and a three-way backend agreement
test. BWM and NWB agreement is shown on `d23a44ef`.

### 2026-09-28 — Session manifest (`neurodecoder/data/manifest.py`), and what BWM's "good units" are

**Decision:** `build_manifest(ephys_root, behaviour_root)` builds two tables
from the metadata of the extracted `bwm_ephys` 1.2.1 and `bwm_behavior` 2.0.0
releases, without loading any session. Both release versions are checked
through their `manifest.json`. `write_manifest` / `read_manifest` store them
as `sessions.parquet`, `insertions.parquet` and `provenance.json`, replacing
the old copy atomically.

- **`sessions`, one row per eid:** subject, lab, date, session number,
  `n_probes`, `n_units` (all clusters), `n_label1_units`, `n_good_units`,
  `n_trials`, `n_included_trials` (`bwm_include`), `regions` (sorted Beryl
  acronyms of the good units) and `modalities`.
- **`modalities` uses canonical names** (`wheel`, `pose_left/right/body`)
  and says what the **`bwm_behavior` release holds**, not what
  `load_session("bwm")` loads: that backend doesn't read behaviour yet.
  Motion energy and pupil aren't in `bwm_behavior` at all.
- **`insertions`, one row per probe:** unit and channel counts, plus the
  probe's **tip** and **top** positions in **meters**, bregma-relative (the
  same frame as the units' `x/y/z`). The tip is the mean position of the
  channels nearest the probe tip (smallest `localCoordinates_y`), and the top
  is the mean of the channels farthest from it. The release's channel table
  has `mlapdv_x/y/z` (µm) and `localCoordinates_x/y`, not the `x/y/z` /
  `axial_um` its docs list.
- **Every count is computed from the underlying tables** and must equal the
  release's own per-session values (`n_insertions`, `n_good_units`,
  `n_trials`, `n_included_trials`), or building raises.
- **No config keys.** The two release folders are arguments, and nothing
  needs to find the manifest through config yet. Keys get added when a
  command-line tool does.

**Finding: BWM's "good units" are not simply `label == 1`.** 75,708 clusters
have `label == 1`, but the release's good-unit table holds 75,395, a strict
subset (none outside it). The 313 dropped label-1 units, spread over 89 of
699 probes:
- **307 are located in `void` (187) or `root` (120)**, i.e. outside the brain
  or without a region. That matches ibl-ai-agent's `INVALID_ACRONYMS` /
  `INVALID_BERYL_ACRONYMS = {"void", "root"}` filter.
- **6 are unexplained:** 3 in SCiw, 2 in CUL4 5, 1 in CENT2. Their Beryl
  mappings are valid (SCm, CUL4 5, CENT2, checked with `iblatlas`), other
  units in the same regions on the same probes were kept, and their label,
  `bitwise_fail`, spike counts, firing rates and presence ratios are normal.
  The reason isn't recorded anywhere I found.

The manifest therefore checks that good units ⊆ `label == 1` (and raises
otherwise), rather than equality. It reports `n_label1_units` next to
`n_good_units` so the gap stays visible. **Phase 2 consequence:** ROADMAP's
planned `qc/units.py` default ("IBL's own QC label, plus firing rate floor,
plus RP ceiling") would keep units that aren't in the brain unless it also
excludes `void`/`root`. Decide that explicitly there. For session `d23a44ef`
the two sets happen to coincide, which is why the earlier NWB-vs-BWM check
matched exactly.

**Verified on the real releases** (building takes about 1 s):
- Totals equal every count verified earlier: 459 sessions, 139 mice, 12
  labs, 699 probes, 75,395 good units, 621,733 clusters, 295,920 trials.
- `d23a44ef`: 2 probes, 1,961 clusters, 398 good units, 410 trials.
- 15 sessions have no pose.
- The tip is deeper than the top on all 699 probes. The median tip-to-top
  span is 3.78 mm, matching Neuropixels 1.0's 3.84 mm recording length.
- 99.98% of good units lie between their probe's tip and top (±10 µm).

### 2026-09-28 — `load_session` entry point and `configs/data.yaml`

**Decision:** `load_session(eid, backend="bwm", *, config=None,
use_cache=True)` in `neurodecoder/data/load.py` is the single way to get a
session, as the ROADMAP Phase 1 interface describes. Each backend is
registered with its data **source** identity and **loader version**, which
together with the eid form the cache key.

| Name | Reads | `source` in the cache key |
|---|---|---|
| `"bwm"` (default) | `bwm_compressed.load_session_bwm` | `{"dataset": "bwm_ephys", "version": "1.2.1"}` |
| `"nwb"` | `dandi_nwb.load_session_nwb`, from a local copy if one exists, else streamed | `{"dandiset": "000409", "version": "0.260309.1324"}` |

- **Paths come from `configs/data.yaml`** (the first file in `configs/`, per
  §7): a `data_root` plus subpaths for the BWM release, the local 000409
  copies and the cache. `NEURODECODER_DATA_ROOT` overrides `data_root`.
  Unknown or missing keys raise, so a typo can't be silently ignored.
- **`LOADER_VERSION = 1`** now sits in each backend module, next to its
  pinned data version. Bump it whenever that backend's mapping into a
  `Session` changes. It's part of the cache key, so the bump invalidates
  that backend's cached sessions, as R6 does for `PREPROC_VERSION`. It lives
  in the backend, not in the registry, so the person changing a mapping sees
  it.
- **Local and streamed NWB share one cache key.** A local copy is the pinned
  version's asset (downloads are SHA-256 checked, and the streamed load is
  tested equal to the local one). Two local copies for one eid raise instead
  of one being picked.
- **`"one"` was added later** (see "ONE backend"), with its pinned sorting and
  trials revisions in the cache key. An unregistered name raises, listing the
  available backends.
- `use_cache=False` bypasses the cache completely: no read, no write.

**Verified:**
- `load_session` returns sessions identical to the direct backend calls for
  BWM and NWB on `d23a44ef`, and the second BWM call is served from the cache
  without calling the backend.
- Config parsing, the env override and key rejection are tested.
- The NWB file resolution is tested: local copy first, streaming otherwise,
  and ambiguous local copies rejected.

### 2026-09-28 — Session cache (`neurodecoder/data/cache.py`)

**Decision:** `SessionCache(root)` stores loaded `Session`s on disk, one
directory per key. `get_or_load(key_parts, loader)` returns a cached session,
or runs the loader and stores its result.

- **Keys are content addresses.** The key is the sha256 of the key parts plus
  `CACHE_FORMAT_VERSION`, independent of dict order. The required parts are
  `eid`, `backend`, `source` (the pinned data version, e.g. BWM `1.2.1` or
  DANDI `000409@0.260309.1324`) and `loader_version`. Changing any of them
  makes a new entry, so a stale session can't be served after a data or
  loader change. Entries live at `root/<first 2 hex chars>/<sha256>`.
- **Where ROADMAP's `(eid, PREPROC_VERSION, config_hash)` fits:** Phase 1
  caches raw sessions, with no preprocessing, so there's no
  `PREPROC_VERSION` yet. When Phase 2 caches binned tensors it adds
  `PREPROC_VERSION` and the config hash as further key parts. The key scheme
  already accepts extra parts.
- **Writes are atomic.** An entry is written to a `.tmp-*` directory next to
  its final location and renamed into place, and a failed write deletes the
  temporary directory. An entry exists only if its `meta.json` does, so a
  crash can never leave a half-written entry that later reads as valid.
- **No pickle.** Spikes are one flat `float64` `.npy` plus per-unit offsets.
  Behaviour series are `.npy` files, loaded with `allow_pickle=False`.
  Units and trials are parquet, and `meta.json` holds the eid, time bounds,
  unit and behaviour order, channel names, capabilities and the key parts.
  Pickle is fast, but it isn't safe to load and is fragile across versions.
- **Loaded entries are re-validated,** because every `Session` checks itself
  when built.
- **`loader_version` comes from the caller.** The cache doesn't import any
  backend; the planned `load_session(eid, backend=…)` entry point will
  supply each backend's loader version. Bump it whenever a backend's mapping
  changes, just as R6 bumps `PREPROC_VERSION`.

**Verified:** 12 tests cover:
- round trips that match in every field and dtype, on a fixture with text
  columns, float32, booleans, NaNs, a named multi-channel series and a unit
  with no spikes;
- the real BWM and NWB `d23a44ef` sessions round-tripping exactly;
- key determinism and required parts;
- a mid-write failure leaving no entry and no temporary directory;
- an entry without `meta.json` counting as a miss;
- a loader returning the wrong eid being rejected.

**Measured on `d23a44ef`:**

| Backend | From source | Cache write | **Cache read** | Entry size |
|---|---|---|---|---|
| BWM compressed | 3.4 s | 0.2 s | **0.2 s** | 198 MB |
| DANDI NWB (local) | 5.9 s | 0.6 s | **0.9 s** | 687 MB |

Both are well under ROADMAP Phase 1's 5 s target. A streamed NWB session
(~20 min) drops to 0.9 s after the first load. Disk cost is roughly 10 GB
per 50 BWM sessions or 34 GB per 50 NWB sessions: NWB includes every cluster
and the video signals. Nothing is compressed, because read speed is the
point. Compression is the lever if disk becomes the constraint.

**Alternatives considered:** pickle (unsafe to load, fragile); HDF5 (would
work, but parquet keeps pandas dtypes and index names without custom code);
hashing the backend's source code into the key (automatic, but any comment
edit would invalidate everything, including 20-minute streamed sessions).

### 2026-09-28 — BWM compressed backend (`neurodecoder/data/backends/bwm_compressed.py`), and `numcodecs`

**Decision:** `load_session_bwm(eid, root)` reads one session from an
extracted `bwm_ephys` release into a `Session`. It adds **`numcodecs`**
(0.16.5; requires only `numpy` and `typing_extensions`) as a dependency,
because the spike shards are `numcodecs` Blosc (zstd, shuffle) arrays,
exactly as ibl-ai-agent writes them.

- **The release version is pinned to `1.2.1`.** `manifest.json` must say
  `bwm_ephys` `1.2.1`, or the backend raises, for the same reason the DANDI
  and ONE revisions are pinned.
- **Spike decoding follows ibl-ai-agent's own reader:** times =
  (`cumsum(delta ticks)` + `time_origin_ticks`) × 100 µs. Shards whose
  `format`, `time_encoding` or `cluster_encoding` differ from the known
  values are rejected. Each shard's per-unit counts must equal its per-spike
  cluster assignments, and they must match the units table's cluster IDs
  and `spike_count`, or the backend raises.
- **Spike-time error:** the encoder rounds (`np.rint`) to 100 µs ticks. The
  shards in this release record origin 0, from an older encoder, so there's
  a single rounding and the error is at most **50 µs**. ibl-ai-agent's
  current encoder rounds the first spike separately, which would allow up
  to 100 µs; that doesn't apply to these shards. Measured on `d23a44ef`
  against NWB: max 50.00 µs, over all 398 units.
- **Unit IDs are `{probe_name}_{cluster_id}`**, the same IDs the NWB backend
  produces, so the two can be compared directly. Units are good units only
  (`label == 1`), and `x/y/z` are BWM's bregma-relative meters, which makes
  BWM the canonical source for coordinates.
- **Trials come from the Brain Wide Map paper's frozen trials table**
  (`provenance.yaml`: `trials_table: bwm_tables/trials.pqt`). All 13
  canonical fields are present, including `stimOff_times`, which
  ibl-ai-agent's docs don't list. `bwm_include` (290 of 410 trials in
  `d23a44ef`) is kept as an extra column; filtering is a later choice.
- **Behaviour is declared missing:** it lives in the separate
  `bwm_behavior` dataset, which isn't loaded yet.

**Cross-backend results against the NWB backend on `d23a44ef`:**
- Exactly the NWB units with `label == 1.0`, with the same IDs.
- Every unit's spike count is identical, and spike times are within 50 µs.
- 12 of 13 trial fields are identical.
- **`firstMovement_times` differs, and structurally.** Both sources put
  movement onsets on a 1 kHz grid, offset by a constant phase (BWM 0.12402 ms;
  ONE's `2025-03-03` revision, which NWB matches, 0.42318 ms). Every
  difference is that phase plus whole samples (−2 in 4 trials, −1 in 290,
  0 in 115, +1 in 1): 405 of 410 trials differ by under 1 ms, and the worst
  is 1.70084 ms. So ONE's 2025-03-03 revision re-extracted movement onsets on
  a shifted resampling grid. **Phase 2 consequence:** a movement-onset target
  can move by up to ~2 ms, occasionally into the next 20 ms bin, depending on
  which backend it's built from. Pick one source for that target and record
  the choice (Phase 2 ADR).

**Performance:** 3.5 s per session. A stable `argsort` on the int64 unit
indices took 14.8 of an initial 16.5 s. Casting to `uint16` (at most 459
good units per insertion in BWM) lets NumPy use its linear-time radix sort,
with an identical order. It falls back to int64 if an insertion ever has
more than 65,535 units.

**Alternatives considered:** Using ibl-ai-agent's package as a dependency
(not on PyPI, and it brings their whole tool stack); `blosc`/`blosc2` instead
of `numcodecs` (the shards are written by `numcodecs`, so its decoder is the
one guaranteed to match).

### 2026-09-27 — Add `remfile` to stream NWB files from DANDI

**Decision:** Add `remfile` (0.1.15; depends only on `h5py`, `numpy`,
`requests`) as a dependency. `load_session_nwb` accepts an `https://` URL as
well as a local path, and streams the URL with `remfile` → `h5py` → `pynwb`,
fetching only the byte ranges it reads. `dandi_asset_url(eid)` resolves an
IBL eid to its `desc-processed` asset's S3 URL through DANDI's REST API (stdlib
`urllib`, so the much heavier `dandi` package isn't needed).

**Why:** ROADMAP Phase 1 says to stream `desc-processed` assets from 000409
via `remfile` + `h5py` + `pynwb` and not bulk-download; ROADMAP §8 lists
`remfile` as the Phase 1 streaming addition.

**Pinned DANDI version:** the resolver reads the published, immutable
version **`0.260309.1324`** (2026-03-09), never the draft, which can change
under us just as ONE's data revisions do. For `d23a44ef` the asset in that
version has the same ID and SHA-256 as the local, checksum-verified file.
Moving to a newer version is a deliberate change, recorded here.

**Verified:** streaming session `d23a44ef` equals the local load in every
field: units, trials, all 1,961 spike trains, all 10 behaviour series and
the capability report (`test_streamed_load_equals_local_load`).

**Cost, measured on this machine:** opening the file takes about 38 s
(pynwb reads the file's structure with many small requests, so it's
latency-bound). A full streamed session load took about **20 minutes**,
versus about 6 s from a local file. `remfile` fetches one range at a time
over one connection, and a single connection to S3 from here gets about
0.3–0.4 MB/s. Streaming is therefore for opening a session once, or for
reading just its metadata or a part of it. Anything repeated should go
through the Phase 1 cache (`data/cache.py`), as the roadmap already says.
`remfile`'s `disk_cache` option and its private `_max_threads` setting could
speed this up, but they aren't used: the first belongs with the cache
design, and the second is a private API.

**Alternatives considered:** the `dandi` package (much heavier, and only
needed here for the asset lookup); downloading whole files (what the
roadmap says not to do across the dataset).

**Consequences:** the two streaming tests only run when
`NEURODECODER_NETWORK_TESTS=1` is set, so CI and routine runs stay offline
and fast. The full comparison test takes about 20 minutes here.

### 2026-09-27 — DANDI NWB backend mappings (`neurodecoder/data/backends/dandi_nwb.py`)

**Decision:** `load_session_nwb(source)` reads DANDI 000409 `desc-processed`
NWB files into a `Session`, from a local path or streamed from a URL (see
"Add `remfile`"). Every mapping was checked value by value against ONE for session
`d23a44ef-1402-4ed7-97f5-47e9a7a504d9` (sorting revision `2024-05-06`, trials
revision `2025-03-03`), with zero differences:

| Canonical | NWB | Check against ONE |
|---|---|---|
| `intervals_0/1`, `stimOn/stimOff/goCue/firstMovement/response/feedback_times` | `start_time`, `stop_time`, `gabor_stimulus_onset/offset_time`, `auditory_cue_time`, `wheel_movement_onset_time`, `choice_registration_time`, `feedback_time` | max \|diff\| = 0 on all 410 trials |
| `choice` | `mouse_wheel_choice`: `clockwise` → **+1**, `counter_clockwise` → **−1** | crosstab exact: 118 / 292, no off-diagonal |
| `feedbackType` | `is_mouse_rewarded` True → +1, False → −1 | exact: 304 / 106 |
| `contrastLeft/Right` | `gabor_stimulus_contrast` (**percent**) split by `gabor_stimulus_side`, ÷100, NaN off-side | exact |
| `probabilityLeft` | `probability_left` | exact |
| `label` | `ibl_quality_score` (0, ⅓, ⅔, 1) | equals ONE `clusters.metrics.label` |
| `depths` | `distance_from_probe_tip_um` | equals ONE `clusters.depths` |
| `firing_rate` | `firing_rate` | exact |

- **Unit ID** is NWB's `unit_name`, e.g. `probe00_0`, i.e. probe plus ONE's
  `cluster_id` (same order as ONE on both probes). `cluster_id`,
  `cluster_uuid` and `location` (full Allen region name, via each unit's
  `max_electrode`) are kept as extra columns for matching against other
  backends. `probe_name` uses IBL's lowercase `probe00`; NWB's column says
  `Probe00`.
- **Unrecognised values raise.** An unknown choice string (e.g. a no-go
  trial, of which this session has none), an unknown stimulus side, or a
  contrast outside IBL's percent set (0, 6.25, 12.5, 25, 50, 100) raises
  instead of being guessed. The percent check also catches a source that
  switches to fractions.
- **`behaviour.wheel` is the raw `WheelPosition`** (radians, 755,552 irregular
  samples), not IBL's smoothed 1 kHz position/velocity. The smoothing filter is
  a preprocessing choice (R6), and ROADMAP Phase 2 warns it changes wheel R².
- **`time_bounds` is the span of all loaded data** (spikes, trial times,
  wheel), because the processed file holds no raw recording to define it. For
  this session that's 0.0008–3668.940 s. The wheel's last sample is 2 ms after
  the last spike, so spike-only bounds would wrongly reject it.
- **Declared missing, with reasons:** `units.acronym` (NWB has full region
  names; mapping names to acronyms isn't implemented), `units.x/y/z` (NWB
  electrode coordinates are Allen CCF µm, and the BWM convention isn't
  verified yet), and `behaviour.lick` (events, not a sampled signal).
- **Per-camera video signals are loaded**, one key per camera, each on its own
  clock:
  - `motion_energy_{left,right,body}` come from
    `motion_energy/{Left,Right,Body}CameraMotionEnergy`.
  - `pupil_{left,right}` come from the **raw** `pupil/{Left,Right}PupilDiameter`,
    not `...Smoothed`, because smoothing is preprocessing (R6). The NaNs in it
    are kept (25 left, 221 right in `d23a44ef`).
  - `pose_{left,right,body}` come from `pose_estimation/{Left,Right,Body}Camera`.
    Each camera's keypoints are stacked into one series with named columns
    `{keypoint}_x`, `{keypoint}_y` (px) and `{keypoint}_likelihood`, keypoints
    in alphabetical order (body 1, left 6, right 11 in `d23a44ef`).
  - The **likelihood is kept unthresholded**, matching ibl-ai-agent's
    `likelihood_thr=0`, because filtering low-confidence points is a
    preprocessing choice.
  - Keypoint names are NWB's series names in snake_case
    (`RightPupilBottom` → `right_pupil_bottom`), taken as the file gives them.
    Matching them to BWM's pose naming is a cross-backend question for later.
- **Checked before stacking:** within each camera, every pose keypoint shares
  exactly the same timestamps, and those equal the camera's motion-energy and
  pupil timestamps. The backend raises if a keypoint is on its own clock, and
  it rejects rate-sampled series without timestamps rather than rebuilding
  them.
- **A signal absent from a file is declared missing** ("no `module/name` in
  this NWB file") instead of raising. About 4% of BWM sessions have no pose,
  per ibl-ai-agent's docs.

**Resolved contract issue:** the first version of this backend couldn't hold
IBL's three cameras (body ~30 Hz, left ~60 Hz, right ~150 Hz, with different
timestamps), because the `Session` contract had one `TimeSeries` per field.
The contract now has one key per camera (see "The `Session` contract"), and
this backend loads all of them.

**Consequences:** Loading this session takes about 6 s, including full
contract validation. The tests needing the real file are skipped in CI (the
file is 1.3 GB and not in the repo); the mapping tests run everywhere.

### 2026-09-27 — The `Session` contract (Phase 1, `neurodecoder/data/session.py`)

**Decision:** Every backend returns a `Session` that validates itself on
construction, so an invalid one can't exist:
- **Canonical names are IBL's ALF names**, as used by the compressed BWM
  (`stimOn_times`, `goCue_times`, `firstMovement_times`, `response_times`,
  `feedback_times`, `intervals_0/1`, `choice`, `feedbackType`,
  `contrastLeft/Right`, `probabilityLeft`, plus `stimOff_times`; units:
  `probe_name`, `acronym`, `x/y/z`, `depths`, `label`, `firing_rate`). Other
  backends (NWB, whose names differ, see `docs/PRIOR_ART.md` §D) map *to* these.
- **Every canonical field must be declared present or missing with a
  reason.** Present means it exists in the data and isn't entirely NaN;
  missing means it's absent (no placeholder column). Unknown field names are
  rejected, which catches typos. This is §7's "missing means missing",
  enforced in code.
- **Times are seconds on the session clock.** Spike times and behaviour
  timestamps must be finite and sorted, and all spike, trial and behaviour
  times must fall within `time_bounds`. A span over 24 h is rejected as
  "probably milliseconds", the trap SpikeLab's loader sets.
- Extra backend-specific columns (e.g. `cluster_uuid`) are allowed.
- **Behaviour signals from video have one key per camera** (revised
  2026-09-27): `motion_energy_left/right/body`, `pupil_left/right`,
  `pose_left/right/body`, plus `wheel` and `lick`. IBL films each session with
  three cameras at different rates (body ~30 Hz, left ~60 Hz, right ~150 Hz
  in session `d23a44ef`), with different timestamps, so a single
  `motion_energy` series can't hold them. The body camera doesn't see the
  pupil, so there's no `pupil_body`. The original single-key names
  (`motion_energy`, `pose`, `pupil`) are now rejected as unknown.
- **Multi-channel series must name their columns.** A `TimeSeries` with
  `(n_samples, n_channels)` data needs `channel_names` (one unique name per
  column), and 1-D data takes none. Per-camera pose is several tracked
  keypoints, each with x and y, and an unlabelled `(n, 10)` array would leave
  "which column is the left paw's y?" to guesswork.

**Why:** Phase 1's cross-backend tests are only meaningful if all backends
produce the same shape of object with the same field names, and if a missing
field can't masquerade as data.

**Alternatives considered:** Validating in a separate function that callers
could forget to run; NWB's column names as canonical (rejected: BWM is the
primary training source).

**Consequences:** The roadmap's interface says `Session.available:
CapabilitySet`; the class is named `Capabilities`, with the same role. What a
unit ID is (`cluster_uuid` vs `(pid, cluster_id)`) is left to each backend.
BWM and NWB were later shown to share `(probe_name, cluster_id)` exactly (see
`docs/PRIOR_ART.md` §D).

**Correction (2026-09-27):** an earlier version of this entry said BWM stores
trial times as float32 and so needs a ~1 ms comparison tolerance. That came
from ibl-ai-agent's schema docs. The extracted `bwm_ephys` 1.2.1
`metadata/trials.parquet` stores them as **double (float64)**, so trial times
can be compared exactly. BWM's per-unit `firing_rate` *is* single precision
(differs from NWB by at most 3.45e-6 on `d23a44ef`), so compare that one with
a tolerance.

### 2026-09-27 — NWB intake reads with `pynwb` directly; SpikeLab is not a dependency

**Decision:** `neurodecoder/nwb/` and the NWB data backend read files with
`pynwb`, which is already in the Fixed stack (§8). SpikeLab is **not** added
as a dependency. If we later want its analysis methods (STTC, slice stacks,
`RateData`), we wrap it by converting our `Session` into a `SpikeData`, and
still don't fork it. That's the Phase 0 task 3 "reuse vs wrap" outcome.

**Why:** This rests on actually loading DANDI 000409's processed NWB for
session `d23a44ef-…` through SpikeLab's `load_spikedata_from_nwb`
(2026-09-27). It works: 1,961 units and 61,981,600 spikes in 15 s, matching
ONE exactly (see `docs/PRIOR_ART.md` §A). But for our purposes:
1. It keeps 5 per-unit attributes out of the units table's 27 columns and
   drops `cluster_uuid` (the key for matching units across backends) and
   every IBL QC column `qc/` needs (`ibl_quality_score`,
   `sliding_rp_violation`, `noise_cutoff`, `presence_ratio`, amplitudes,
   drift).
2. It fills gaps silently: duration is inferred from the last spike and a
   missing start time becomes 0.0. §7 says missing means missing.
3. Spike times are in milliseconds; our interface (ROADMAP Phase 1) uses
   seconds, so every call would need a conversion that's easy to get wrong.
4. It reads spikes only, no trials or behaviour, which the same file holds
   (trials table; `wheel`, `motion_energy`, `pose_estimation`, `pupil`,
   `lick_times` processing modules).

A few lines of `pynwb` read all of it.

**Alternatives considered:** Wrapping SpikeLab's loader and reading the
missing columns separately (two readers for one file, for no gain); forking
it (ruled out by CLAUDE.md §2).

**Consequences:** One fewer dependency. The "prefer wrapping SpikeLab" rule in
§2 still applies to *analysis*; this decision covers *intake* only.

### 2026-09-27 — Downloaded data lives in `~/data/neurodecoder`, outside the repo

**Decision:** All downloaded datasets live under `~/data/neurodecoder/`, not
in the git repo:

| Path | What | Source | Checksum |
|---|---|---|---|
| `bwm_compressed/archives/bwm_ephys-1.2.1.tar` | ibl-ai-agent compressed BWM, spikes/units | `ibl-brain-wide-map-public` S3, `resources/ibl-agent-data/` | SHA-1 `b18c5c7a2944be510800010eb3df90aac84a2a52` |
| `bwm_compressed/archives/bwm_behavior-2.0.0.tar` | ibl-ai-agent compressed BWM, behaviour | same | SHA-1 `1c37dd1c38d46ec80067c8a25772dfe2468a1ce1` |
| `dandi/000409/sub-DY-016/…_ses-d23a44ef-…_desc-processed_behavior+ecephys.nwb` | DANDI 000409 processed NWB, same session as the NEDS run | DANDI asset `ecf201ee-c535-4371-a2bd-b6da93c0fbb5` | SHA-256 `563b902729aab23bf6ce3457396629521dcd2ffe9244e551ce46bf16c940dab4` |

Checksums come from ibl-ai-agent's `scripts/download_datasets.py` and DANDI's
asset metadata, and each download is verified against them.

**Why:** About 10.5 GB of archives plus about 12 GB extracted shouldn't sit
inside the repo tree (git operations, editor indexing, backups). A fixed path
recorded here means future sessions and forks find the data rather than
downloading it again.

**Alternatives considered:** A gitignored `datasets/` folder in the repo.
Also running ibl-ai-agent's own installer script, rejected because it writes
config into their repo; plain `curl` plus checksum verification is more
transparent.

**Consequences:** Phase 1's `data/backends/bwm_compressed.py` and the Phase 1
cache should take this root from config under `configs/`, not hard-code it.
Note that ibl-ai-agent's downloader fetches **`bwm_ephys` 1.2.1**, while
their `docs/bwm/README.md` still describes 1.2.0. Trust the extracted
dataset's own schema/version files over their README.

### 2026-09-27 — CI installs CPU-only PyTorch

**Decision:** `.github/workflows/tests.yml` installs `torch` from PyTorch's CPU
wheel index before `pip install -e ".[dev]"`.

**Why:** The default PyPI `torch` wheel on Linux bundles CUDA libraries
(about 2 GB). CI has no GPU and our Phases 0–4 need none, so CPU wheels keep
every run fast without changing which package versions get resolved.

**Consequences:** When GPU-dependent tests arrive (Phase 5+), they need a
separate job or marker; this job stays CPU-only.

### 2026-09-27 — NEDS's loader depends on a Hugging Face org that is now empty

**Decision:** Patched `load_ibl_dataset` in
`external/NEDS/src/utils/dataset_utils.py` (its only
`get_user_datasets(...)` call, line 209) to list the local `*_aligned`
directories under `cache_dir` instead of querying Hugging Face. Same
`org/eid_aligned` format, so the rest of the loader is unchanged. It is a
local patch to a gitignored clone.

**Why:** `get_user_datasets` calls `datasets.list_datasets()`, which pages
through every public dataset on Hugging Face, then filters to
`ibl-repro-ephys/`. The list is used only to check that the eid is
"published". The data itself is then read from local disk
(`load_from_disk(f"{cache_dir}/{eid}_aligned")`), i.e. whatever NEDS's own
`prepare_data.py` wrote. Two failures stack up:
(1) paging all of Hugging Face without logging in hits `HTTP 429 Too Many
Requests`; (2) even with no rate limit, a targeted query
(`HfApi().list_datasets(author="ibl-repro-ephys")`) returns **0 datasets**, so
the check would raise `ValueError: ... not found in the user's datasets`.
`create_dataset.py` and `train.py` both go through this loader, so NEDS's
training pipeline cannot run for an outside user without this patch, whatever
the package versions.

**Alternatives considered:** Pinning versions (doesn't help: the dependency is
on an external org's contents, not a package API); requesting access to the
org (unknown if it still exists privately).

**Consequences:** The data being trained on is exactly what `prepare_data.py`
produced locally, which was verified: prep's train/val/test rows
(249/36/72) match `create_dataset.py`'s written files exactly. Fourth NEDS
environment break so far; each has been a distinct cause (native build,
`ibllib` API, macOS `spawn`, Hugging Face org), not the same one repeating.

### 2026-09-27 — NEDS data prep loops forever on macOS unless forced to `fork`

**Decision:** Patched `external/NEDS/src/prepare_data.py` to call
`multiprocessing.set_start_method("fork", force=True)` right after its stdlib
imports. Local patch to a gitignored clone, as with the `SessionLoader` fix.

**Why:** `src/utils/ibl_data_utils.py` creates a `multiprocessing.Pool` in
three places (lines 201, 458, 491), even with `n_workers=1`, and
`prepare_data.py` has no `if __name__ == "__main__":` guard. macOS defaults
to the `spawn` start method, so every pool worker re-imports and re-executes
the whole script: it re-downloads session data, fails to start its own pool
(`RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase`), dies, and the parent
pool respawns it. **The parent never exits.** One run did this silently for
2.5 hours (532 `RuntimeError`s in its log), downloading into the same cache
directory a later run was using. NEDS was developed on Linux SLURM clusters,
where the default is `fork`, so this never shows up there.

**Alternatives considered:** Adding a `__main__` guard (more invasive, since
the whole script body is top-level code); running on Linux (not available).

**Consequences:** Anyone reproducing NEDS on macOS needs this patch. More
generally: **a crashing child process does not mean the job exited.** Before
relaunching any background data job, check with `pgrep` that the previous
one is gone, and write the exit code to the log instead of trusting a piped
command's status (`cmd | tail` reports `tail`'s exit code, which is how the
first `SessionLoader` crash showed up as "exit 0"). Data written while two
runs shared a cache directory was discarded and re-downloaded, not reused.

### 2026-09-27 — `ONE-api`/`ibllib` installs on this machine need llvmlite/numba pinned first

**Decision:** Before installing `ONE-api`, `ibllib`, or anything that pulls in
`numba` (directly or via `iblutil`), run
`pip install "llvmlite==0.44.0" "numba<0.61.3,>=0.60"` first, in whatever venv
you're targeting.

**Why:** `pip install ONE-api` (and separately, installing NEDS's `env.yaml`
deps in a plain venv) fails building `llvmlite` from source — `numba`'s latest
version pulls `llvmlite>=0.49`, which has no prebuilt wheel for macOS x86_64+
Python 3.11 on PyPI (source build needs a matching LLVM toolchain we don't
have). `llvmlite==0.44.0` / a compatible `numba<0.61.3` do have prebuilt
wheels for this platform. Installing them first satisfies the pin before pip
tries to build the newer, wheel-less version as a transitive dependency.

**Alternatives considered:** Building LLVM via Homebrew to satisfy the source
build — much heavier, not attempted.

**Consequences:** Applies to this machine's `.venv` (project) and to
`external/NEDS/.venv` (gitignored, for the Phase 0 reproduction attempt).
Anyone re-running either install on similar hardware will hit the same failure
without this pin.

### 2026-09-27 — NEDS's `SessionLoader` call is incompatible with current `ibllib`

**Decision:** Patched both `SessionLoader` call sites in
`external/NEDS/src/utils/ibl_data_utils.py` (lines 107 and 263:
`SessionLoader(one, eid=eid)` → `SessionLoader(one=one, eid=eid)`) to keep the
Phase 0 reproduction attempt moving. This is a local patch to a gitignored
external clone, not a change to our own code. The first attempt patched only
line 107. Line 263 runs inside a pool worker and crashed the next run, so when
fixing an API break, grep for every call site first. The other IBL calls in
that file (`SpikeSortingLoader`, `BrainRegions`, `get_spike_counts_in_bins`)
were checked against `ibllib` 4.0.1 and are compatible.

**Why:** `external/NEDS` (cloned per the Phase 0 audit) pins `ibllib`
unversioned in `env.yaml`, so a fresh install pulls current `ibllib` (4.0.1).
Current `ibllib`'s `SessionLoader` is a dataclass with `one` as keyword-only;
NEDS's code passes it positionally, which now raises
`TypeError: SessionLoader.__init__() takes 1 positional argument but 2 ... were given`.
This is exactly the "NEDS environment resolution" failure point
ROADMAP.md Phase 0 predicted in advance.

**Alternatives considered:** Pinning `ibllib` to an older, contemporaneous
version instead of patching the call site — not attempted yet; would be the
more faithful fix if more breaks of the same kind turn up, since patching
call sites one at a time doesn't scale if the API drift is broader than this
one call.

**Consequences:** Confirms Phase 0 task 1's expected failure mode. If more
`ibllib`/`iblatlas`/ONE-api API breaks surface while trying to reproduce a
NEDS number, switch strategy to pinning old versions of the whole IBL stack
rather than continuing to patch individual call sites.

---

### 2026-09-27 — Phase 9 (future-horizon prediction) stays at full scope

**Decision:** Do not cut ROADMAP.md Phase 9 to a stretch goal. Keep it at its
original 30–60 h scope.

**Why:** Phase 9 is explicitly conditional on auditing SpikeProphecy
(arXiv 2605.12992) first. Cloned and read its source
(`external/SpikeProphecy`, gitignored): it forecasts future *neural population
spike counts* from past spikes, evaluated with a population-similarity metric
decomposition — a different task from Phase 9's future-*behaviour* decoding
evaluated against an autocorrelation-of-behaviour baseline. Its only
behaviour-decoding code is a same-timestep auxiliary classification head in an
Appendix C distillation experiment, not a t+100/250/500 ms forecast. See
`docs/PRIOR_ART.md` §E for the full audit.

**Alternatives considered:** Cutting Phase 9 to the small controlled
experiment the roadmap describes as the fallback — rejected because the
condition that triggers that fallback ("SpikeProphecy already covers this")
is false.

**Consequences:** Phase 9, if reached, still needs its own behaviour-forecast
implementation and its own autocorrelation baseline; nothing from
SpikeProphecy is directly reusable for it. `src/data/ibl_behavior_loader.py`'s
IBL trial-field extraction pattern may still be worth a look when we write
`neurodecoder/targets/`, independent of the forecasting question.

### 2026-09-27 — Pin local venv to Python 3.11 via Homebrew

**Decision:** Installed `python@3.11` via Homebrew (`/usr/local/opt/python@3.11`)
and recreated `.venv` with that interpreter, matching the `>=3.11,<3.12` pin in
`pyproject.toml`.

**Why:** The system Python was 3.13, which is outside the Fixed stack's pin
(§8). Installing `python@3.11` required first updating Xcode Command Line
Tools (26.2 → 26.6), which needed the user's own admin auth and was done by
the user, not this session.

**Alternatives considered:** Relaxing the pyproject.toml Python pin to allow
3.13; using pyenv instead of Homebrew.

**Consequences:** `.venv` now resolves to Python 3.11.16. `pytest` and
`pre-commit run --all-files` both re-verified passing under it.

## Template

### YYYY-MM-DD — Short title

**Decision:**

**Why:**

**Alternatives considered:**

**Consequences:**
