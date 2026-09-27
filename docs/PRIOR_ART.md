# Prior art — verified

Status of each claim: **[V]** verified against repo/paper text, **[A]** to be
audited by you in Phase 0 by actually running the code.

---

## A. SpikeLab (braingeneers/SpikeLab)

van der Molen et al., bioRxiv 2026, doi 10.64898/2026.04.25.720833.

**[V]** A text-to-analysis framework for spike data. Composable data structures
(`SpikeData` holding per-unit spike times in ms; `RateData` for binned rates;
slice stacks for event-aligned analysis) confirmed directly in
`src/spikelab/spikedata/{spikedata,ratedata,spikeslicestack,rateslicestack}.py`.
Loads HDF5, NWB, KiloSort/Phy, SpikeInterface (`data_loaders/data_loaders.py`).
Exports to KiloSort and NWB (`data_loaders/data_exporters.py`). Ships an MCP
server (`mcp_server/`). Includes spike-sorting pipelines — Kilosort2 (MATLAB),
Kilosort4 (PyTorch), rt-sort — and a Kubernetes batch-job CLI
(`spikelab-batch-jobs`), all confirmed present in `src/spikelab/spike_sorting/`
and `src/spikelab/batch_jobs/`.

**[V, reworded] "Bounded autonomy" is our paraphrase, not SpikeLab's own term** —
no such phrase appears in the repo. What is verbatim in
`skills/spikelab-analysis-implementer/SKILL.md`: a **"Correctness over
efficiency"** section ("Always prioritize faithfully executing the user's
request over minimizing computation time... Do not silently reduce data
windows, downsample, skip units, coarsen bin sizes"); a **"Strict Boundary
Rules"** section restricting the agent to a user-named analysis directory and
forbidding edits to library code; a mandate to **use SpikeLab's own methods
instead of reimplementing analyses** ("Do not implement custom neuroscience
analysis logic... outside of the library"); and repeated **clarification-before-
acting** instructions in both the analysis-implementer and spikesorter skills.
The behaviors are real; the label is ours.

**[A, not verified by code]** The four-task LLM benchmark and the
mouse-Neuropixels / human-Utah-array / organoid-MEA validation claim comes from
the bioRxiv preprint text, not from anything in the cloned repo — there is no
benchmark harness or results file in the source tree to run. Leaving this as
**[A]**: verifying it means reading the paper's methods/results, which is a
different task than running the code.

**[Corrected] "It contains no behaviour decoder" is wrong.**
`src/spikelab/spikedata/decoding.py` exists and is non-trivial: cross-validated
classifier decoding (`RidgeClassifier`, `LogisticRegression`, `MLPClassifier`,
`RandomForestClassifier` from sklearn) of categorical labels from a per-slice
response-amplitude matrix `(n_slices, n_units)`, with cross-entropy, confusion
matrices, regularization sweeps, latency-dependent decoding, and drift/novelty
measures (`temporal_decoding_decay`, `novelty_per_group`,
`distinctness_per_group`). But it is scoped narrowly and differently from what
we need: its docstrings and variable names (`fit_model_stim_response.py`-style)
show it decodes **stimulus identity** from **evoked responses within one
session/preparation** — not cross-animal behaviour decoding. Cross-validation is
leave-one-out or k-fold *within* one dataset; there is no session- or
animal-held-out split anywhere in the module. "Probabilities" come straight from
`predict_proba` or a softmax over `decision_function` — useful for a relative
log-loss, explicitly documented as **not a calibrated probability**. No
temperature scaling, no conformal sets, no ensembling. It is also not currently
referenced by any of the agent skill files, so an agent using SpikeLab
conversationally would not be steered toward it. Net correction: **no
cross-animal model, no calibrated uncertainty layer** (both still true) — but
**"no decoder at all" overstates it; a narrower, single-session, uncalibrated
stimulus decoder exists.**

**[V] Loaded a real IBL NWB file through it** (Phase 0 task 3, 2026-09-27):
`load_spikedata_from_nwb` on DANDI 000409's
`sub-DY-016_ses-d23a44ef-…_desc-processed_behavior+ecephys.nwb` (1.32 GB,
SHA-256 matching DANDI). It worked in 15 s: 1,961 units (674 on `Probe00`,
1,287 on `Probe01`), 61,981,600 spikes, 61.1 min, 30 distinct Allen region
names. **But it keeps only 5 per-unit attributes** (`electrode`,
`group_name`, `location`, `location_label`, `unit_id`) out of the file's 27
units columns. It drops `cluster_uuid` and all IBL QC metrics, and fills in a
missing duration or start time silently. Decision: NWB intake uses `pynwb`
directly and SpikeLab isn't a dependency (see `docs/DECISIONS.md`).

- **Reuse:** `SpikeData`/`RateData` for analysis, if we ever need their
  methods (wrap our `Session` into them). **Not** their NWB loader for intake
  (see above). The MCP server pattern; the skill-file design patterns above
  (not the "bounded autonomy" phrase), the closest published precedent for
  our §6 LLM boundary. Possibly `decoding.py`'s CV/log-loss plumbing as a reference for
  our own `evaluation/`, though ours needs session/animal-level splits it does
  not have.
- **Modify:** nothing yet. Wrap, don't fork.
- **Do not copy:** the spike-sorting and Kubernetes layers. Out of scope.

## B. ibl-ai-agent (int-brain-lab/ibl-ai-agent)

**[V]** A repository you clone and open a coding agent inside (Codex or Claude
Code); it supplies instructions, skills and references so the agent can write
analysis code, plot, and publish a report. Explicitly scientist-in-the-loop, not
one-shot. Authors include Rossant, Chapuis, Paninski, Raiser, Winter, Harris.

**[V]** Critically for us: it uses a **compressed BWM representation**, downloaded
from a public archive into `reports/datasets/`, made of two datasets on by
default (confirmed from `docs/bwm/README.md` in the cloned repo, not just the
top-level README):

- **`bwm_ephys`** (v1.2.0) — public archive **6.03 GB**, 4,215 files. 139 mice,
  459 sessions, 699 insertions, **75,395 good units** (621,733 rows in the full
  `clusters.pqt`, matching §D's brain-wide total), 267,264 channels, 295,920
  trials, 2,066,041 events, **4,152,659,397 spikes** written across per-insertion
  blosc shards. Also ships per-unit peak-channel waveforms and log-binned
  autocorrelograms. Spike times at 0.1 ms resolution, as claimed.
- **`bwm_behavior`** (v2.0.0) — public archive **2.9 GB** (5.2 GB on disk), 480
  files, same 459 sessions / 295,920 trials. Wheel data for all 459 sessions;
  pose for 444/459 (96.7%), preferring Lightning Pose over DeepLabCut per camera
  when both exist (measured: LP for 436/437 left-camera, 432/432 right-camera,
  253/260 body-camera sessions; only 8 camera-instances are DLC-only). Ships
  precomputed trial/wheel/pose/event-aligned feature tables plus
  `movement_state_epochs` and `quiescence_state_epochs` parquets — i.e. some
  target engineering (movement-state discretization) is already done for us
  here, not something we need to invent from scratch in Phase 2.
- **`bwm_lfp`** (v1.0.0, opt-in, not downloaded by default) — 14 GB, all 699
  probe recordings in one HDF5, 384 ch (695 recordings) or 96 ch (4 NP2.4
  recordings), 250 Hz (decimated from 2500 Hz).

**Corrected:** "under 10 GB" is the sum of the two *default* datasets' archive
sizes (6.03 + 2.9 ≈ 8.9 GB), not a bound on everything BWM-related — the LFP
dataset alone is 14 GB and is correctly excluded from that figure. The agent
falls back to the ONE API for anything else.

**[V]** Explicitly scoped to preprocessed IBL spikes/task/video — not raw
Neuropixels, not general neurodata.

- **Reuse:** the compressed BWM representation is your Phase 1 shortcut. Use it
  for model development; treat NWB as an *inference-time* concern, not a
  training-time one.
- **Modify:** their skill/report contracts are worth reading as a template for
  `agent/`.
- **Do not copy:** their analysis-agent framing as your product. Yours is a
  trained model with a reliability layer; theirs is an exploration tool.

## C. NEDS (arXiv 2504.08201, github.com/yzhang511/NEDS, ibl-neds.github.io)

This is the one that constrains your novelty. Read it before designing anything.

**[V]** Multimodal, multi-task transformer unifying encoding and decoding via
multi-task masking (neural, behavioural, within-modality, cross-modal).
Confirmed from `src/prepare_data.py` (`binsize: 0.02`, i.e. **20 ms bins**) and
`src/multi_modal/encoder_embeddings.py`, `src/models/stitcher.py`:

- **Tokenization is per-timestep, not per-unit.** Each 20 ms bin's *entire*
  population spike-count vector (fixed at `n_channels` = the training config's
  unit count, e.g. 668) is one token: linearly projected
  (`token_embed` → activation → `projection`) up to `hidden_size`. There is no
  per-(unit, time-bin) token — this is a **fixed population matrix**, exactly
  the design the roadmap's Phase 5 avoids ("fixed population matrices break on
  unit-count change"). Sequence length is capped at `max_F=100` timesteps (2 s
  of context at 20 ms).
- **Cross-session unit-count variation is handled by per-session "stitching"
  layers** (`StitchEncoder`/`StitchDecoder` in `stitcher.py`), not by the shared
  transformer: each `eid` gets its own `nn.Linear` mapping *that session's*
  neuron count into the shared fixed-size space and back out. These per-session
  matrices are exactly what ROADMAP.md Phase 5 flags as "a large share of its
  parameters [that] do not transfer for free" — confirmed in code, not just
  inferred from the paper.
- **Session embeddings are a plain lookup table**
  (`nn.Embedding(len(eid_lookup), hidden_size)` in `EncoderEmbeddingLayer`),
  indexed by an `eid → index` dict built from `data/train_eids.txt` +
  `data/test_eids.txt` at import time. They contain no session content (no
  metadata, no computed statistic) — just a learned per-session vector. **A
  session not in those two files has no session embedding at all**; NEDS has no
  mechanism to infer one for a genuinely new session. This directly confirms
  ROADMAP.md Phase 5's failure point and means "zero it / average it / infer it"
  is a real design decision we still have to make, not a solved problem.
- Adds **temporal** (position) and **modality** embeddings alongside the session
  embedding, all summed into one `x_embed` added to the projected spike token.

**[V]** Trained on the IBL **repeated site** dataset: 83 mice, 5 regions, 10 labs,
standardized pipelines. Up to 73 training animals, **10 held out**, 74 training
sessions.

**[V]** Beats linear regression, multi-session reduced-rank regression, POYO+ and
NDT2 across whisker motion, wheel velocity, choice, and block prior. Multi-session
NEDS beats everything; single-session beats everything except on block decoding.

**[V]** ~3M params single-session encoder, ~12M multi-session, ~150M including
session-specific input matrices and linear decoders. ~13M tokens from ~14 h of
data.

**[V]** Emergent property: learned embeddings predict brain region without being
trained to.

**[V]** Code public; `train_eids.txt` / `test_eids.txt` define the splits.

**What it does NOT do:** calibrated probability outputs, conformal or ensemble
uncertainty, OOD detection, session-level trust gating, selective prediction,
arbitrary-NWB intake, or a capability report. It assumes a well-formed IBL
session and returns a point estimate.

**[V] Confirmed in code (`src/multi_modal/mm.py`, `self.mod_loss`):** losses are
`PoissonNLLLoss` for spikes, plain `MSELoss` for wheel/whisker (continuous),
and `CrossEntropyLoss` for choice/block (discrete). No learned variance, no
logvar head, no ensembling, no dropout-at-inference, no temperature-scaling
step anywhere in `src/models/` or `src/multi_modal/`. `ModelOutput`
(`src/models/model_output.py`) carries only `loss` and `n_examples` — there is
no field for a predictive distribution. This is not a gap we might have missed;
the architecture has nowhere to put uncertainty. Confirms the original claim
precisely rather than just repeating it.

**[V] Data prep, from actually running `src/prepare_data.py`** on held-out test
session `d23a44ef-1402-4ed7-97f5-47e9a7a504d9` (2026-09-27, after the
environment patches in `docs/DECISIONS.md`):

- **Trial-aligned windows, not continuous time.** Each sample is one trial,
  from −0.5 s to +1.5 s around `stimOn_times`, in 100 bins of 20 ms. This
  session: 410 trials × 100 bins × 1,961 units before filtering.
- **Unit filter uses a 5 Hz floor, not 0.2 Hz.** The config says
  `fr_thresh: 0.2`, but the code keeps units with mean rate `> 1/fr_thresh`,
  i.e. 5 Hz. 884 of 1,961 units survived. **The filter runs on all 410 trials
  before the split**, so test trials take part in unit selection. It's
  unsupervised (no labels used), so the leak is mild, but our `qc/` must not
  do this (R3).
- **The within-session split breaks our split rules.** Trials are shuffled at
  random (`np.random.choice`, global `np.random.seed(42)`) and sliced
  70/10/20 into train/val/test. That keeps whole trials together, which our
  "trial-level targets split on trials" rule allows. But it **interleaves
  trials with no temporal blocking and no gap**, against the Blocking and Gap
  rules in `SPLITS_AND_LEAKAGE.md`. The neighbours of each test trial,
  including trials in the same IBL block, sit in the training set. **Expected
  consequence (not measured yet):** NEDS's single-session numbers, block prior
  especially, should come out higher than our `ceiling_within` computed under
  our rules. Don't read a gap between the two as our model underperforming.
- **Revision drift: ONE silently loads the newest data revision.** On disk:
  spike sorting is `#2024-05-06#` (before the paper); the trials table,
  including the `stimOn_times` every window is aligned to, is `#2025-03-03#`
  (one month before the April 2025 preprint); motion energy for the left,
  right and body cameras is `#2025-05-30#`, `#2025-05-31#` and `#2025-06-02#`
  (**after** the preprint). NEDS's whisker-motion-energy target was therefore
  built from different files than a fresh download gets today, and the
  alignment times may differ too. **An exact match to a published NEDS number
  is not guaranteed even with correct code.** A mismatch may come from data
  revisions, not our setup. For our Phase 1 cache key, see ROADMAP.md
  Phase 1: "dataset revisions change under you". This is a concrete instance
  of it.

**[V] Training, from a 1-epoch smoke test** (same session, 2026-09-27):

- **The full pipeline runs end to end** after four local patches (see
  `docs/DECISIONS.md`): `prepare_data.py` → `create_dataset.py` → `train.py`
  → per-modality validation metrics and checkpoints. Real exit code 0.
- **On this Mac it ran on Apple's GPU (MPS), not the CPU.** PyTorch/accelerate
  picked `mps:0` automatically. The paper used CUDA, so small numerical
  differences are possible.
- **Cost:** 16 training steps per epoch (249 training trials, batch 16), about
  1 s per step once warmed up (the first step takes 28 s). The default is 2,000
  epochs with evaluation after every epoch, so roughly **11–14 hours per
  session** on this machine. That's an estimate: evaluation and checkpoint
  saving weren't timed separately.
- **The 1-epoch metrics are chance level and must not be quoted** (choice
  0.50, block 0.33, negative wheel/whisker R²). They show the pipeline runs,
  nothing about performance.
- **Validation (used for checkpoint selection) uses balanced accuracy**
  (`src/trainer/base.py:490`); evaluation computes both plain and balanced
  accuracy (`src/utils/eval_utils.py:464–465`). The paper text says
  "classification accuracy". When comparing against our contract (§5 requires
  balanced accuracy), use NEDS's balanced-accuracy output explicitly.

**[V, partly via paper summary] A single-session reproduction of a published
number isn't possible as specified.** From the arXiv HTML (checked
2026-09-27): single-session results are reported **only as averages and
distributions over the 10 held-out sessions**. Figure 2B shows a per-session
scatter plot, but the sessions aren't identified by eid. The reported numbers
also come from a random hyperparameter search over **50** models (the repo's
`train.sh` defaults to 30). The repo contains **no linear or reduced-rank
baseline code**, so the paper's baseline numbers can't be regenerated from
it either. Approximate averages read off Figure 2A by a summarisation tool
were **not** recorded here because they aren't verified.

- **Reuse:** their session-level split files (`train_eids.txt`/`test_eids.txt`,
  instant comparability) but **not** their within-session trial split (see
  above: random and interleaved, no gap), their 20 ms tokenization,
  their session-embedding trick, their baselines as your baselines.
- **Modify:** their decoder heads to emit distributions rather than points.
- **Do not copy:** the pretraining scale. You cannot match it and do not need to.

## D. IBL Brain Wide Map

**[V]** 621,733 neurons across 279 brain areas; standardized visual
decision-making task (wheel turn to centre a stimulus); 19 labs. Published
milestone results 2025.

**[V]** Available three ways: ONE API (`ibllib`), the compressed representation in
ibl-ai-agent, and NWB on DANDI as **dandiset 000409**, converted by
catalystneuro/IBL-to-nwb using NeuroConv. Both raw ephys and processed
behaviour/spike-sorting files exist; `desc-processed` assets carry units without
raw traces.

**[V]** Streaming works and is cheap. A SpikeInterface PR reports building a
curatable analyzer from one streamed 000409 processed file (898 units, 20.7M
spikes) in ~14 s transferring ~22 MB, with the ~130 MB spike read deferred.

**Implication:** do not plan a bulk download. Stream, cache the binned tensors.

**[V] Backend agreement, first data point (2026-09-27).** For session
`d23a44ef-1402-4ed7-97f5-47e9a7a504d9`, ONE (spike sorting revision
`2024-05-06`) and DANDI 000409's processed NWB agree **exactly**: 1,961
clusters (674 + 1,287) and 61,981,600 spikes (20,767,948 + 41,213,652). That
is zero-tolerance agreement on counts, as the Phase 1 test requires. Facts
Phase 1 must design around:

- **Both ONE and the NWB hold all Kilosort clusters, not just IBL's good
  units.** ibl-ai-agent's compressed BWM holds good units only (75,395 of
  621,733 brain-wide), so its unit counts will differ **by design**. Compare
  backends on a common unit set, matched by `cluster_uuid` (an NWB units
  column), rather than expecting the raw counts to match. How IBL's "good"
  label maps onto NWB's `ibl_quality_score` / `kilosort2_label` still needs
  checking against the compressed data once it's extracted.
- **NEDS uses all clusters too:** its pre-filter count for this session is
  also 1,961, so it applies no IBL QC label, only its 5 Hz firing-rate floor
  (§C). Our Phase 2 `qc/units.py` plans to use the IBL label as well, so
  NEDS's unit set is looser than ours. Keep that in mind when comparing.
- **The NWB units table has 27 columns** including `cluster_uuid`,
  `ibl_quality_score`, `sliding_rp_violation`, `noise_cutoff`,
  `presence_ratio`, `kilosort2_label`, `firing_rate`, amplitudes, drift,
  `distance_from_probe_tip_um` and `waveform_mean`.
- **Trial column names differ between IBL's own backends.** NWB uses
  `gabor_stimulus_onset_time` (ONE: `stimOn_times`), `mouse_wheel_choice`
  (ONE: `choice`), `probability_left`, `block_type`, `block_index`,
  `wheel_movement_onset_time`, `feedback_time`, etc. Behaviour lives in NWB
  processing modules `wheel`, `motion_energy`, `pose_estimation`, `pupil`,
  `lick_times`, `passive_protocol`. So Phase 1 needs a column mapping layer
  even between IBL sources, not only for foreign NWB (Phase 11).

## E. Adjacent work you must not be surprised by

- **POYO / POYO+** (Azabou et al.) — multi-session, multi-task decoding with
  unit-identity-free tokenization. Already solves "neuron 1 ≠ neuron 1".
- **NDT2** (Ye et al.) — multi-session masked modelling for spikes; session/subject
  context embeddings.
- **SpikeProphecy** (arXiv 2605.12992, NeurIPS 2026 D&B track;
  `github.com/JohnMinnick/SpikeProphecy-...`, cloned to `external/SpikeProphecy`)
  **[V] — audited, does NOT cover our §3C.** It forecasts **future neural
  population activity itself** (autoregressive spike-count prediction, 50 ms
  bins), not future *behaviour*. 105 sessions (39 Steinmetz + 66 IBL repeated
  site, ~89,800 neurons), seven architecture baselines (Mamba, HGRN2,
  GatedDeltaNet, LRU, Transformer, LSTM, RSynaptic SNN), evaluated with a
  population-metric decomposition (temporal fidelity, spatial pattern accuracy,
  magnitude-invariant alignment) — no behaviour-forecasting metric anywhere.
  The only behaviour-decoding code (`src/data/ibl_behavior_loader.py`,
  `src/distill/multi_head_loss.py`) is a **same-timestep** auxiliary
  stimulus/choice classification head used to regularize the Appendix C
  SNN-distillation experiment, trial-masked to active trials — it decodes the
  present, not t+100/250/500 ms, and is not evaluated against an
  autocorrelation-of-behaviour baseline.
  **Roadmap implication:** ROADMAP.md Phase 9 is conditional on this audit
  ("if SpikeProphecy already covers this, reduce to a small controlled
  experiment... cut §3C to a stretch goal"). It does not cover it.
  **Phase 9 should stay at its original 30–60 h scope, not be cut.**
  Incidental find worth reusing: `tests/test_data/test_*leakage*.py` includes a
  "PopGLM-as-leakage-catch" test — a concrete pattern for R1/R2-style leakage
  tests worth looking at when writing `splits/guards.py` in Phase 2.
- **Conformal prediction** is mature and model-agnostic, with split-CP requiring
  only a held-out calibration set. It has been applied to neural decoding
  (e.g. ConformalHDC) and to brain-to-text, where CTC-trained decoders were shown
  to be systematically over-confident. You are applying a known method to a new
  setting, not inventing one.

---

## Comparison matrix

| | SpikeLab | ibl-ai-agent | NEDS | POYO+/NDT2 | **This project** |
|---|---|---|---|---|---|
| Problem | agentic spike analysis | agentic IBL exploration | encode+decode at scale | multi-session decoding | trustworthy cross-animal decoding |
| Input | sorted spikes, many formats | compressed BWM | IBL repeated site | multi-session spikes | BWM (train) + arbitrary NWB (infer) |
| Neural repr. | SpikeData/RateData | raw spike times 0.1 ms | 20 ms tokens + session emb. | unit tokens | 20 ms tokens + unit-metadata tokens |
| Cross-session | n/a | n/a | **yes** | **yes** | yes (reuse) |
| Cross-animal | n/a | n/a | **yes, 10 held out** | yes | yes (reuse) |
| Behaviour decoding | stimulus-ID only, single-session | ad hoc | **yes** | yes | yes (reuse) |
| Latent state | descriptive tools | no | implicit embeddings | implicit | explicit, descriptive only |
| **Uncertainty** | no | no | **no** | no | **core** |
| **OOD / gating** | no | no | **no** | no | **core** |
| **Arbitrary NWB** | loads NWB | no | no | no | **core** |
| Interpretability | analysis-level | agent prose | region-predictive embeddings | limited | attribution + gating rationale |
| Agent layer | **skills, MCP, bounded autonomy** | **skills, reports** | no | no | reuse both patterns |
| Reproducibility | good | good | split files public | varies | strict (see CLAUDE.md) |

---

## Honest novelty assessment

**Not novel, do not claim:**
- Permutation-invariant / set-based neuron representation — POYO, NDT2, NEDS.
- Session embeddings for cross-session transfer — NEDS, NDT2.
- Multi-task shared backbone with multiple heads — NEDS.
- Multi-scale temporal binning — standard.
- Cross-animal pretraining and fine-tuning — NEDS, with held-out animals.
- Neural population forecasting (predicting future *spikes* from past spikes)
  — benchmarked, SpikeProphecy. Note this is a different task from Phase 9's
  behaviour forecasting, which SpikeProphecy does not benchmark (see §E) — so
  Phase 9 behaviour-at-a-future-horizon decoding, evaluated against an
  autocorrelation-of-behaviour baseline, remains open, not "already done
  elsewhere."
- Conformal prediction, temperature scaling, deep ensembles, MC dropout — all
  off-the-shelf.
- Agent-assisted scientific interpretation of spike data — SpikeLab, ibl-ai-agent.

**Plausibly novel, in descending order of defensibility:**

1. **Session-level OOD gating with demonstrated selective-prediction benefit.**
   Claim shape: *for a cross-animal decoder on IBL, a representation-space
   distance score computed at session level predicts per-session decoding error,
   and abstaining on the worst-scoring x% of sessions reduces error by y% more
   than abstaining on the lowest-softmax-confidence x%.* Nobody has shown this on
   IBL. It requires no new architecture. It is falsifiable in one experiment.
   **This is your minimum viable research result.**

2. **A calibration audit of large-scale neural decoders.** Claim shape: *NEDS and
   the linear baselines are over/under-confident by this much on held-out
   animals, and post-hoc method Z fixes it.* Cheap, useful, and a natural paper —
   the brain-to-text over-confidence finding suggests the result will be positive.

3. **A capability-report NWB intake layer + benchmark harness.** Genuinely
   missing infrastructure, low scientific novelty, high community value. Best
   framed as a tools paper, not a discovery.

4. **Everything else in your spec.** Treat as engineering, not contribution.

**The framing that survives review:** "We do not propose a new architecture. We
show that existing large-scale decoders are miscalibrated on held-out animals,
and that a session-level distribution-shift score recovers reliability." That is
a real, modest, publishable result. Anything grander needs evidence you do not
have yet.
