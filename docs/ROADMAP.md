# Roadmap

Reordered from the original 13-phase spec. Two structural changes:

- **Latent-state modelling moved from Phase 7 to Phase 10.** It is the easiest
  place to fool yourself and contributes nothing to the core claim.
- **Arbitrary-NWB intake moved from Phase 11 to Phase 3 (probe only) and Phase 11
  (full).** You need to know early whether NWB intake is a weekend or a month,
  because it changes whether the project has two deliverables or one.

Hours are **solo developer with Claude Code**, including debugging, failed runs,
and writing things down. They are not optimistic.

---

## Phase 0 — Prior-art audit and scaffold

**Objective.** Convert `docs/PRIOR_ART.md` from claims into verified facts, and
stand up a repo that enforces the rules in CLAUDE.md.

**Tasks.**
1. Clone and *run* NEDS. Reproduce one decoding number from the paper on one
   session. Record what broke.
   **Amended 2026-09-27:** closed as a negative result. The pipeline runs end
   to end after four patches, but the paper publishes only 10-session averages
   (with a 50-model hyperparameter search) and no baseline code, so there is
   no per-session number to reproduce. The numeric comparison moves to Phase
   3. See `docs/NEGATIVE_RESULTS.md`.
2. Clone ibl-ai-agent. Download its compressed BWM representation. Confirm size
   and contents.
3. Clone SpikeLab. Load one NWB file through it. Decide reuse vs. wrap.
4. Find and audit SpikeProphecy. Determine whether your §3C future-prediction
   goal is already benchmarked. **If it is, cut §3C to a stretch goal.**
5. Scaffold the repo per CLAUDE.md §9. `pyproject.toml` with pinned deps,
   pre-commit (ruff + black), pytest, CI running tests on push.
6. Write `docs/DECISIONS.md` (ADR-style, one entry per non-obvious choice) and
   `docs/NEGATIVE_RESULTS.md` (empty, with a template).

**Outputs.** `docs/PRIOR_ART.md` with every `[A]` promoted to `[V]` or corrected.
~~One reproduced number from NEDS.~~ NEDS's pipeline run end to end, with its
breakages recorded (amended 2026-09-27, see task 1). A repo that passes
`pytest` on an empty suite.

**Success criteria.** You can state, in one sentence each, what NEDS does that
you will reuse and what it does not do that you will add. If you cannot, you are
not ready for Phase 1.

**Failure points.** NEDS environment resolution; IBL credential setup; the
temptation to skip the reproduction step. Do not skip it — everything downstream
assumes you can compare against a real number.

**Hours: 25–45.**

---

## Phase 1 — Data access and session manifest

**Objective.** One function that returns a session's spikes, units, metadata and
behaviour, from any of three backends, with caching.

**Tasks.**
- `data/manifest.py` — build a session table: eid, subject, lab, date, probe
  count, insertion coordinates, region list, unit count, QC-pass unit count,
  trial count, available behavioural modalities. Persist as parquet.
- `data/backends/one_backend.py` — ONE API access.
- `data/backends/bwm_compressed.py` — the ibl-ai-agent representation.
- `data/backends/dandi_nwb.py` — stream `desc-processed` assets from dandiset
  000409 via `remfile` + `h5py` + `pynwb`. Do not bulk download.
- `data/cache.py` — content-addressed cache keyed on
  `(eid, PREPROC_VERSION, config_hash)`.

**Interface.**
```python
load_session(eid, backend="bwm") -> Session
# Session.spikes: dict[unit_id, np.ndarray]     spike times, seconds
# Session.units:  pd.DataFrame  (n_units, ...)  region, depth, hemisphere,
#                                                x/y/z, amp, contamination, ...
# Session.trials: pd.DataFrame  (n_trials, ...) stim on/off, choice, feedback,
#                                                contrast, block, response time
# Session.behaviour: dict[str, TimeSeries]      wheel, pose, pupil, lick
# Session.available: CapabilitySet
```

**Tests.** Three backends return byte-identical unit counts and
spike-count totals for the same eid (tolerance zero on counts; document any
legitimate discrepancy as an ADR). Timestamps monotonic. Trial times within
session bounds. Cache hit returns identical arrays.

**Success criteria.** `load_session` works for 50 sessions across ≥10 subjects
and ≥3 labs, from cache, in under 5 s each.

**Failure points.** This is the phase people underestimate by 3×. Timestamp
alignment between spikes, trials and video is where real bugs live. ONE
credentials and dataset revisions change under you. Cross-backend discrepancies
are common and *must* be resolved, not averaged over.

**Hours: 50–90.**

---

## Phase 2 — Preprocessing, targets, and the split registry

**Objective.** Deterministic tensors and leakage-proof partitions.

**Tasks.**
- `qc/units.py` — unit filtering. Default: IBL's own QC label, plus firing rate
  floor, plus refractory-period violation ceiling. Every threshold in config.
- `preprocess/binning.py` — spike times → `(n_units, n_bins)` counts at
  configurable width. Implement 10/20/50 ms; default 20 ms for NEDS comparability.
- `preprocess/normalize.py` — per-unit statistics **fit on train split only**,
  serialized into the artifact.
- `preprocess/windows.py` — sliding windows → `(n_windows, n_units, n_context_bins)`.
- `targets/` — wheel velocity (continuous), choice (binary, trial-level),
  movement onset (event), block prior (binary), plus a `movement_state`
  discretization matching your §1 example output. Document every threshold.
- `splits/registry.py` — see `docs/SPLITS_AND_LEAKAGE.md`. Emits a JSON split
  file with a hash. **Nothing downstream accepts a session list directly.**
- `splits/guards.py` — assertion helpers imported by every training entry point.

**Shapes.** Single session: `X (n_units, n_bins)`, `y (n_bins,)` or
`(n_trials,)`. Batched: `X (B, n_units, n_context)`, unit metadata
`(B, n_units, d_meta)`, mask `(B, n_units)`, `y (B,)` or `(B, n_horizon)`.

**Tests.** `test_no_temporal_leakage` — train and test windows share no bin
index, with a gap ≥ context length. `test_no_subject_leakage` — subject sets are
disjoint in animal-held-out splits. `test_normalization_fit_on_train_only` —
mutating test data does not change stored statistics. `test_binning_determinism`
— same input, same `PREPROC_VERSION`, byte-identical output.

**Success criteria.** All six split types (§ SPLITS doc) generate and validate.
Round-trip a session to tensors and back to spike counts within float tolerance.

**Failure points.** Wheel velocity is the classic one — differentiating a noisy
encoder signal, and the filter choice materially changes decoding R². Fix the
filter, record it as an ADR, never tune it against a test metric. Trial-level
targets replicated across bins create implicit leakage if trials straddle a split
boundary: split on trials, not bins, for trial-level targets.

**Hours: 60–110.**

---

## Phase 3 — Baselines, evaluation harness, and an NWB probe

**Objective.** The first real number, and an early read on NWB cost.

**Tasks.**
- `models/baselines/` — ridge regression (wheel velocity), logistic regression
  (choice, movement state), Poisson GLM, multi-session reduced-rank regression.
- `evaluation/metrics.py` — balanced accuracy, AUROC, F1, log loss, ECE, R²,
  correlation, MAE, RMSE. Per-session, always.
- `evaluation/contract.py` — implements the six-row table in CLAUDE.md §5.
  Running a model without it should be awkward.
- `evaluation/nulls.py` — circular-shift null, trial-structure-only baseline.
- `nwb/probe.py` — **thin**: open an arbitrary NWB, report what it finds
  (units table, spike_times, electrodes, region column, trials interval, named
  behavioural series, timestamps, rates). No normalization, no model. Run it
  against 000409 files and 3–5 unrelated dandisets.
- `cli/evaluate.py`.
- **NEDS comparison (moved from Phase 0, 2026-09-27).** Run the baselines on
  NEDS's 10 held-out test sessions (`external/NEDS/data/test_eids.txt`) and
  compare against the paper's reported 10-session averages. Use balanced
  accuracy explicitly for choice/block. Remember that NEDS's within-session
  split is random-interleaved with no gap, so its numbers are expected to sit
  above our `ceiling_within`. See `docs/PRIOR_ART.md` §C.

**Success criteria.** Wheel-velocity R² and choice accuracy on a within-session
split that are in the same ballpark as published IBL linear baselines. The
evaluation contract table prints for every run. The NWB probe classifies ≥5
foreign files without crashing.

**This is the decision gate.** If baselines do not clear `null_trialstruct` by a
convincing margin, something in Phases 1–2 is wrong. Stop and find it. Do not
proceed to deep models.

**Failure points.** Choice decoding is heavily confounded by trial structure and
by stimulus contrast — the `null_trialstruct` row exists precisely for this.
Expect your first "great" accuracy to be the null.

**Hours: 45–80.**

---

## Phase 4 — Generalization protocol

**Objective.** Turn "cross-session generalization" from an aspiration into a
measured quantity for the *baselines*, before any deep model exists.

**Tasks.** Run the full evaluation contract under each split type: within-session,
held-out session same animal, held-out animal, held-out lab, held-out region,
held-out probe configuration. Produce a generalization gap table.
`evaluation/protocols.py`, `viz/generalization.py`.

**Success criteria.** A single table showing, per split type, baseline
performance and its drop from the within-session ceiling. **This table is the
motivation for the rest of the project.** If the drop is small, your central
premise is weak and you should say so publicly.

**Failure points.** Held-out region is subtle — regions are not independent
across insertions, and "held out" needs a precise definition (see SPLITS doc).
Held-out lab has very few groups; report it with appropriate humility.

**Hours: 30–55.**

---

## Phase 5 — Population encoder

**Objective.** A unit-identity-free encoder. **Reuse NEDS's design; do not invent.**

**Recommended architecture:** tokens per (unit, time-bin) with learned unit-metadata
embeddings (region, depth, hemisphere, normalized coordinates, log firing rate,
waveform width, QC metrics), plus temporal and session embeddings, into a
transformer encoder with masking over units. Permutation invariance comes from the
absence of positional encoding over the unit axis, not from a DeepSets layer.

**Why not the alternatives:** fixed population matrices break on unit-count
change, which is the whole problem. DeepSets is permutation-invariant but discards
pairwise structure that matters for population codes. Graph representations
require a neuron-adjacency prior you do not have. Set Transformer is closest and
is essentially what attention-over-units already gives you.

**Why not Mamba/SSM first:** the sequence lengths here (context of 25–50 bins at
20 ms) are short. SSMs pay off at long sequence lengths you do not have. Revisit
only if you move to minute-scale context.

**Tests.** `test_permutation_invariance` — shuffling the unit axis leaves output
unchanged to float tolerance. `test_variable_unit_count` — 200 and 900 units both
run. `test_masking` — masked units contribute zero gradient.

**Success criteria.** Beats `baseline_ridge` on held-out **animal** (not just
held-out session). If it only wins within-session, it is not earning its
complexity.

**Failure points.** Session embeddings are unavailable for a genuinely new
session — decide now whether you infer them, zero them, or average them, and
measure the cost of each. NEDS's session-specific input matrices are a large
share of its parameters and do not transfer for free.

**Hours: 80–140.**

---

## Phase 6 — Calibration and uncertainty

**Objective.** Outputs that are probabilities, not scores.

**Tasks.** `uncertainty/` — temperature scaling on a held-out calibration split;
deep ensembles (5 members; the most reliable and the most expensive); MC dropout
(cheap, weaker); split conformal prediction sets for classification and
conformalized quantile regression for wheel velocity; predictive entropy.
`evaluation/calibration.py` — ECE, MCE, reliability diagrams, coverage vs.
nominal for conformal sets, per split type.

**Success criteria.** ECE below 0.05 on held-out animals for at least one method.
Conformal coverage within 2% of nominal on held-out sessions — and **a documented
failure of coverage under animal shift**, which is expected, since exchangeability
breaks. That failure is a finding, not a bug.

**Failure points.** The calibration split must be disjoint from both train and
test at the *session* level, which costs you sessions. Conformal guarantees
assume exchangeability; across animals they do not hold, and reporting them as if
they do would be the single most damaging error you could make.

**Hours: 50–90.**

---

## Phase 7 — OOD detection and selective prediction — *the contribution*

**Objective.** Demonstrate that a session-level trust score works.

**Tasks.**
- `ood/scores.py` — Mahalanobis distance in encoder representation space to the
  training distribution; k-NN distance; energy score; ensemble disagreement;
  metadata-space distance (region composition, unit count, depth profile, firing
  rate distribution, probe coordinates).
- `ood/gating.py` — map score to abstain / low / moderate / high confidence.
- `evaluation/selective.py` — risk-coverage curves, AURC, per-session error vs.
  score correlation.

**The experiment that defines this project:**
> For held-out animals, compute per-session OOD score and per-session decoding
> error. Report Spearman correlation. Then build risk-coverage curves for
> abstention by OOD score vs. by softmax confidence vs. random. Report AURC for
> each.

**Success criteria.** Spearman |ρ| > 0.5 between session OOD score and session
error, and OOD-based abstention beating confidence-based abstention on AURC.
**If both fail, that is still a publishable negative result** — write it up in
`NEGATIVE_RESULTS.md` and say so out loud.

**Failure points.** Too few held-out sessions for a stable correlation — you need
≥20, ideally ≥40, so plan your splits in Phase 2 with this in mind. OOD score
correlating with unit count rather than with genuine distribution shift is the
obvious confound; control for it explicitly.

**Hours: 60–100.**

---

## Phase 8 — Report, CLI, and agent layer

**Objective.** `neurodecoder analyze recording.nwb` producing the output in your §23.

**Tasks.** `cli/analyze.py`; artifact writers for `predictions.parquet`,
`uncertainty.parquet`, `qc.json`, `model_metadata.json`, `report.html`,
`figures/`; `agent/` with the LLM boundary from CLAUDE.md §6 and
`tests/test_agent_no_invention.py`.

**Success criteria.** End-to-end on a fresh session with no manual steps. Every
numeral in the HTML report traceable to an artifact field.

**Hours: 40–70.**

---

## Phase 9 — Future-horizon prediction *(conditional — audit SpikeProphecy first)*

If SpikeProphecy already covers this, reduce to a small controlled experiment:
decode behaviour at t+100/250/500 ms and report performance **relative to the
autocorrelation baseline** — i.e. predicting future behaviour from *current
behaviour* alone. On a structured task like IBL, that baseline is strong and most
apparent "prediction" is trial-structure inference. Say so in the output.

**Hours: 30–60** (or 0 if cut).

---

## Phase 10 — Latent states *(descriptive)*

HMM and switching linear dynamical systems over the encoder representation.
Validated only by: cross-session state-count stability, behavioural correspondence
(mutual information with observed behaviour), transition-timing alignment to task
events. Named `latent_state_k` throughout. Never described as engagement,
preparation, or any cognitive term in code or automated output.

**Hours: 50–90.**

---

## Phase 11 — General NWB compatibility

Full intake: schema probe → capability report → available heads → graceful
degradation. Tested against ≥15 dandisets from different labs and conversion
pipelines.

**Failure point.** NWB is a container standard, not a content standard. Region
labels, wheel encoding and trial columns differ per lab. You will need a mapping
layer and it will never be complete. Design for "I can predict A and B, not C,
and here is why" rather than for universality.

**Hours: 70–130.**

---

## Phase 12 — Benchmarking and write-up

Full matrix across split types, ablations, comparison against NEDS on its own
splits, negative results, paper draft.

**Most valuable ablations, in order:** (1) with/without unit metadata — tests
whether anatomy carries transferable information; (2) fixed-ordering vs.
permutation-invariant — directly tests the cross-animal premise; (3) number of
training animals — the scaling curve reviewers will ask for; (4) with/without
session embeddings at test time; (5) 20 vs 50 ms bins. The rest are nice to have.

**Hours: 60–110.**

---

## Timeline

| Phase | Objective | 10 h/wk | 20 h/wk | 40 h/wk | Depends on |
|---|---|---|---|---|---|
| 0 | Audit + scaffold | 3–5 wk | 1.5–2.5 wk | 1 wk | — |
| 1 | Data access | 5–9 wk | 2.5–4.5 wk | 1.5–2.5 wk | 0 |
| 2 | Preprocess + splits | 6–11 wk | 3–5.5 wk | 1.5–3 wk | 1 |
| 3 | Baselines + eval | 4.5–8 wk | 2.5–4 wk | 1–2 wk | 2 |
| 4 | Generalization protocol | 3–5.5 wk | 1.5–3 wk | 1 wk | 3 |
| 5 | Population encoder | 8–14 wk | 4–7 wk | 2–3.5 wk | 4 |
| 6 | Calibration | 5–9 wk | 2.5–4.5 wk | 1.5–2.5 wk | 5 |
| 7 | OOD + selective | 6–10 wk | 3–5 wk | 1.5–2.5 wk | 6 |
| 8 | CLI + agent | 4–7 wk | 2–3.5 wk | 1–2 wk | 7 |
| 9 | Horizons *(cond.)* | 3–6 wk | 1.5–3 wk | 1–1.5 wk | 5 |
| 10 | Latent states | 5–9 wk | 2.5–4.5 wk | 1.5–2.5 wk | 5 |
| 11 | General NWB | 7–13 wk | 3.5–6.5 wk | 2–3.5 wk | 8 |
| 12 | Benchmark + paper | 6–11 wk | 3–5.5 wk | 1.5–3 wk | 11 |

**Cumulative milestones:**

| Milestone | Phases | Hours | 10 h/wk | 20 h/wk | 40 h/wk |
|---|---|---|---|---|---|
| A. Minimum viable prototype | 0–3 | 180–325 | 4.5–8 mo | 2–4 mo | 1–2 mo |
| B. Research-grade MVP | 0–4 | 210–380 | 5–9 mo | 2.5–4.5 mo | 1.5–2.5 mo |
| C. Strong cross-session decoder | 0–5 | 290–520 | 7–12 mo | 3.5–6 mo | 2–3 mo |
| D. Uncertainty + OOD result | 0–7 | 400–710 | 10–17 mo | 5–8.5 mo | 2.5–4.5 mo |
| E. Full system + agent + NWB | 0–12 | 650–1180 | 16–28 mo | 8–14 mo | 4–7 mo |

Milestone E at 10 h/week is **two to two and a half years**. That is the honest
number. The spec as written is a lab programme, not a side project. Plan to stop
at D and write it up — D is the scientifically meaningful result, and E is mostly
engineering that adds no claim.

Add 15–25% if you have not worked with ONE or pynwb before, and another 15% if
GPU access is intermittent.

---

## Compute

| | Spec | Note |
|---|---|---|
| Minimum dev | 16 GB RAM, 8 cores, 500 GB SSD, no GPU | Phases 0–4 need no GPU. Baselines are sklearn. |
| Recommended dev | 32–64 GB RAM, 1 TB NVMe | Cached binned tensors for 100 sessions at 20 ms are tens of GB, not TB. |
| Minimum GPU | 12 GB (RTX 3060 / 4070) | Enough for single-session and small multi-session encoders. |
| Recommended GPU | 24 GB (4090 / A5000) or rented A100 40 GB | Needed only for Phase 5+ multi-session training and Phase 6 ensembles (5× cost). |
| Storage | 50–150 GB working | The compressed BWM is under 10 GB; stream NWB rather than downloading 000409 in bulk. |

**Staged strategy.** Phases 0–4 on a laptop. Phase 5 on a single consumer GPU
with 20–30 sessions. Only rent cloud GPU once the encoder beats ridge on
held-out animals at small scale — that is the point where scaling is a known-good
investment rather than a guess. Expected training time: baselines, seconds to
minutes; single-session encoder, under an hour; 70-session multi-session encoder,
single-digit GPU-hours per run on a 24 GB card, times however many runs your
hyperparameter search needs — budget 10–20× a single run.

**Do not download the full IBL dataset.** You will not need raw traces at any
point in this roadmap.
