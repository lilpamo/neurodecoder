# Phase 3 gate: methods proposals

**Adopted 2026-09-29: (a), B1 and B2**, as recorded in `docs/DECISIONS.md`. The
blocked scheme for per-bin targets (last paragraph) was not adopted. The text
below is the proposal as written before that decision.

Written 2026-09-29, **after** the first table
(`runs/20260929T070943Z_phase3_first_table`). Any change
that is adopted gets a `docs/DECISIONS.md` entry saying it was made after the
first table, and the gate call rests on the confirmation set
(`configs/runs/phase3_confirmation.yaml`), which no diagnostic has touched.

## (a) Choice and block: "spikes alone vs task" or "task + spikes vs task alone"

**What the current row tests.**
- `model` decodes from spikes only.
- `null_trialstruct` decodes from task variables only: for choice, the current
  signed contrast and block prior plus 10 trials of history; for block, the
  history only.
- The verdict asks whether spikes alone predict better than task variables
  alone.

**What the rules say the row is for:**
- **CLAUDE.md §5:** `null_trialstruct` is "prediction from trial time / task
  variables only, no spikes", and "a result that does not beat
  null_trialstruct is not decoding".
- **docs/SPLITS_AND_LEAKAGE.md, route 1:** "a model given enough temporal
  context can infer trial phase, and from phase infer the likely behaviour,
  with no neural information. `null_trialstruct` exists to catch this."
- **The split doc's smell test:** "Does it beat `null_trialstruct`? If not, you
  decoded the task, not the brain."

**The principle.** The row exists to stop task structure from being credited
as neural information. The two comparisons answer different questions:
- **Spikes alone vs task alone** asks whether spikes are a *better predictor
  than* the task variables.
  - When the target is largely set by variables the experimenter controls, a
    spikes-only decoder can carry real neural information and still lose,
    because the null is handed those variables directly. Choice follows the
    stimulus contrast, and block is revealed by the stimulus history.
  - So a pass here proves neural information, but a fail doesn't prove its
    absence.
- **Task + spikes vs task alone** asks whether neural activity *adds*
  information beyond trial structure. If adding spikes to the task features
  improves prediction on held-out trials, the brain carries information about
  the variable that the task doesn't give away, which is exactly the smell
  test's question.

**Proposal.**
1. **Add a row, replace none.** `model_with_task` is the model under test,
   given its spike features **and** the `null_trialstruct` features. It uses the
   same split, the same test samples and the same gapped training CV. Every
   existing row and verdict stays as it is.
2. **Two penalties, not one.** The two feature groups need separately tuned
   penalties, both chosen by training CV. Otherwise ~100–400 spike features
   under one shared λ can drown ~30 task features, and "adding spikes" would
   look harmful for reasons of fitting, not information.
3. **Verdicts.** Print both:
   - "model vs null_trialstruct" (spikes alone vs task, as now);
   - "model_with_task vs null_trialstruct" (the incremental test).

   Both use the same one-sided paired Wilcoxon over test sessions on the
   primary metric.
4. **The gate** rests on the incremental verdict for every target, because it
   is the question §5 and route 1 ask. The spikes-alone verdict stays in every
   table as the stricter, descriptive result; passing it remains the stronger
   claim, and the report says which one passed.

**Risks.**
- **Changing the gate's criterion after seeing results** is itself a degree
  of freedom ("leakage through you", split doc route 4). So adopt it only on
  the principle above, record it as post-first-table, and make the gate call
  on the confirmation set.
- **The incremental model has more parameters.** Held-out scoring and
  CV-tuned penalties stop that from inflating test scores beyond noise, but 10
  sessions give the paired test limited power: report effect sizes alongside
  p.

**Rejected:** weakening or replacing `null_trialstruct`, or removing features
from it. That is ruled out, and it would hide the task confound rather than
measure around it.

## (b) Block under the within-session split

Block switches among usable block trials (`bwm_include`, probabilityLeft ≠ 0.5)
in the first table's split:

| Session | Train trials / blocks | Test trials / blocks (switches) | Left-block fraction in test | Blocks in session |
|---|---|---|---|---|
| 03d9a098 | 246 / 10 | 60 / 3 (2) | 0.55 | 12 |
| 0841d188 | 169 / 6 | 76 / 3 (2) | 0.68 | 8 |
| 3638d102 | 258 / 10 | 70 / 3 (2) | 0.84 | 12 |
| 4b7fbad4 | 406 / 12 | 98 / 3 (2) | 0.62 | 14 |
| 687017d4 | 182 / 7 | 35 / 3 (2) | 0.20 | 9 |
| 9b528ad0 | 293 / 7 | 85 / 4 (3) | 0.67 | 10 |
| 9fe512b8 | 338 / 10 | 108 / 4 (3) | 0.51 | 13 |
| d23a44ef | 168 / 5 | 55 / 2 (1) | 0.60 | 6 |
| db4df448 | 178 / 6 | 36 / 1 (0) | 0.00 | 6 |
| dfd8e7df | 220 / 6 | 82 / 3 (2) | 0.71 | 9 |

**The problem.**
- **Test blocks hold 1–4 blocks** (median 2 switches), and one session holds a
  single block, where AUROC is undefined.
- **With two or three switches, block labels in the test set are nearly a
  function of time.** A decoder that separates a few contiguous stretches of
  trials, for instance by tracking slow drift, scores well or badly for
  reasons unrelated to any block representation.
- **The circular-shift null inherits the problem.** Shifts of ≥ 100 trials
  preserve the block statistics, but they're scored on the same few-switch
  test sets.

**Options.**
- **B1. Leave-one-block-out with gaps: a new within-session split kind, R2
  compliant.**
  - Each fold holds out one block (or one run of consecutive blocks) as test,
    with a gap of at least max(context, 2 s) before and after it (trials in the
    gaps excluded).
  - It trains on all other blocks, before and after, and fits the normaliser
    per fold on training data only.
  - Every block is tested once, so 6–14 blocks and 5–13 switches per session
    reach the per-session metric, and no session is left with one class.
  - Costs:
    - one fit per block;
    - a change to `splits/`, which CLAUDE.md §10 requires escalating;
    - drift between a block and its neighbours is still present (though
      neighbours carry the *opposite* label, so drift works against the
      decoder rather than for it).
- **B2. A pseudo-session null, as the Brain Wide Map paper uses.**
  - Generate many synthetic block sequences from the task's own generative
    process: a 90-trial unbiased start, then alternating blocks with the
    protocol's length distribution.
  - Decode each with the same pipeline and the same neural data, and place
    the real score in that per-session distribution.
  - Pseudo-blocks share the real blocks' temporal statistics but not their
    identity, so the null absorbs slow drift and the few-switch problem.
  - It is added as its own row (e.g. `null_pseudosession`), never replacing
    `null_shuffle`.
  - Costs:
    - IBL's block generator (brainbox), or a faithful port tested against the
      release's block-length distribution;
    - 100–200 pseudo-sessions × fits per session.

**Recommendation: B1 and B2 together, for block only.**
- **B1 fixes the measurement:** enough switches per session for the metric to
  be defined and stable.
- **B2 is the principled null** for an autocorrelated, experimenter-generated
  label, and it makes block comparable with the Brain Wide Map.
- **Either alone leaves a gap.** B2 on the current split still scores on 1–4
  blocks, and B1 without B2 leaves the drift question to the circular shift.
- **Scope:** choice, wheel velocity and movement state stay on the current
  within-session split under this proposal.

**Related open question, not proposed here.** Step 3 found that the late
test block is a less engaged state for wheel velocity (fewer trials per
minute, less movement). A blocked within-session scheme like B1, with test
blocks spread through the session, would address that too, if you want it
considered for per-bin targets.
