# Prior art — verified

Status of each claim: **[V]** verified against repo/paper text, **[A]** to be
audited by you in Phase 0 by actually running the code.

---

## A. SpikeLab (braingeneers/SpikeLab)

van der Molen et al., bioRxiv 2026, doi 10.64898/2026.04.25.720833.

**[V]** A text-to-analysis framework for spike data. Composable data structures
(`SpikeData` holding per-unit spike times in ms; `RateData` for binned rates;
slice stacks for event-aligned analysis) plus a skill-based agent system the
authors describe as enforcing bounded autonomy — mandatory use of expert-vetted
methods, correctness over efficiency, clarification-seeking on ambiguous
requests. Loads HDF5, NWB, KiloSort/Phy, SpikeInterface. Exports to KiloSort and
NWB. Ships an MCP server. Includes spike-sorting pipelines (Kilosort2/4, rt-sort)
and Kubernetes batch submission.

**[V]** Validated by benchmarking LLMs with and without the library on a
four-task analysis pipeline, March 2026. Demonstrated across mouse Neuropixels,
human Utah arrays, and human forebrain organoid MEA data using the same code path.

**It contains no behaviour decoder, no cross-animal model, no uncertainty layer.**

- **Reuse:** `SpikeData`/`RateData` as the internal spike representation; their
  loaders; the MCP server pattern; above all, the bounded-autonomy skill design —
  it is the closest published precedent for our §6 LLM boundary.
- **Modify:** nothing yet. Wrap, don't fork.
- **Do not copy:** the spike-sorting and Kubernetes layers. Out of scope.

## B. ibl-ai-agent (int-brain-lab/ibl-ai-agent)

**[V]** A repository you clone and open a coding agent inside (Codex or Claude
Code); it supplies instructions, skills and references so the agent can write
analysis code, plot, and publish a report. Explicitly scientist-in-the-loop, not
one-shot. Authors include Rossant, Chapuis, Paninski, Raiser, Winter, Harris.

**[V]** Critically for us: it uses a **compressed BWM representation** — spike
times of all high-quality neurons at 0.1 ms resolution, basic metadata including
brain location, plus behavioural traces (stimulus/response events, wheel, video
keypoints). All BWM experiments fit in **under 10 GB**. The agent falls back to
the ONE API for anything else.

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
Modality-specific tokenizers convert spike counts and continuous behaviour into
**20 ms tokens**; discrete behaviours become repeated token sequences. Adds
**temporal, modality, and session embeddings**.

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

- **Reuse:** their split files (instant comparability), their 20 ms tokenization,
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

## E. Adjacent work you must not be surprised by

- **POYO / POYO+** (Azabou et al.) — multi-session, multi-task decoding with
  unit-identity-free tokenization. Already solves "neuron 1 ≠ neuron 1".
- **NDT2** (Ye et al.) — multi-session masked modelling for spikes; session/subject
  context embeddings.
- **SpikeProphecy** (arXiv 2605.12992) **[A]** — a large-scale benchmark for
  autoregressive neural population *forecasting*, evaluated on Steinmetz 2019
  (39 sessions, 10 mice) and IBL repeated site (66 sessions, up to 1,998
  simultaneous neurons), with 50 ms-binned tensors released. **Audit this before
  you write a line of forecasting code** — it likely covers your §3C.
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
| Behaviour decoding | no | ad hoc | **yes** | yes | yes (reuse) |
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
- Neural population forecasting — benchmarked (audit SpikeProphecy).
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
