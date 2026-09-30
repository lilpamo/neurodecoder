# Neurodecoder Studio: next steps

Proposal, 2026-09-30. Follows the prototype on `studio-prototype`
(`docs/DECISIONS.md`, "Direction change: Neurodecoder Studio"). Each step keeps
the prototype's rule: the UI computes nothing itself. It loads through `data/`
and `qc/`, gets every number from `neurodecoder/analysis/` and draws through
`viz/`.

## 1. Import Kilosort / Phy folders — done (2026-09-30)

Built as planned, with these differences (`docs/DECISIONS.md`, "Phy import and
Phy QC"):
- **Code:** `data/backends/phy.py` (`load_session_phy`), `qc/phy.py` and
  `configs/qc_phy.yaml`. The group is Phy's `good`/`mua`/`noise` label, kept as
  `phy_group`, with `group_file` recording where it came from.
- **Group:** read from `cluster_group.tsv`, else `cluster_KSLabel.tsv`, else
  missing. `cluster_info.tsv` is not read.
- **Depth:** the y position of the peak channel of the cluster's most-used
  template, not an amplitude-weighted position. It is declared missing without
  the template files.
- **Events CSV:** canonical trial column names, and it must include
  `intervals_0` and `intervals_1`. Events outside the span of the recorded
  spikes are refused.
- **Phy QC:** group in `[good]` and a task-period rate of at least 0.1 Hz. IBL's
  sliding refractory-period test was added later (see step 3).
- **Opening data:** through server flags (`--phy FOLDER --events CSV`). The
  folder-picker question is still open.

## 2. Atlas and 3D view — built (2026-09-30)

Built as planned below (`docs/DECISIONS.md`, "Atlas and 3D view, built"), with no
chart library. Two things differ from the plan:
- At Beryl, `root` also holds units that IBL labelled only with a coarse parent
  region (MY, CB, TH), not just fibre tracts.
- The probe strip's region runs span recorded units, so they are not histological
  boundaries.


**What:** where each unit is, at the region level the user picks, in a 3D brain
and along the probe. Plus a visual redesign of the app.

**Region level** (`analysis/atlas.py`):
- A selector for Allen, Beryl or Cosmos. **Default: Beryl.**
- Names, colours and hierarchy come from `iblatlas.regions.BrainRegions`, already
  a dependency: `acronym2acronym(..., mapping=...)`, `rgb`, and parents and
  descendants.
- Remapping is shown, never hidden. At Beryl and Cosmos, fibre tracts and some
  nuclei map to `root` (in d23a44ef, `ml` becomes `root`). Those units are
  labelled "no Beryl region" and counted, not dropped.
- Unit QC keeps using the Allen acronym, so QC doesn't change with the display
  level (R6).

**Hierarchical region filter:**
- A tree built from the iblatlas hierarchy, trimmed to the regions this session
  has units in, with unit counts per node.
- Selecting a node includes its descendants. The heatmap title names the node
  and its count.

**3D brain** (`studio/static/`, three.js):
- The whole-brain outline plus the meshes of the regions present, in their Allen
  colours.
- Each probe track, fitted to its channel or unit positions.
- The selected unit's site as a marker. Clicking a site selects that unit.
- **Coordinates:** IBL `x, y, z` (metres, from bregma) are converted to CCF µm
  in `analysis/atlas.py` with iblatlas's own landmarks, not with constants
  copied into the code. The page receives CCF coordinates and computes none.
- **Meshes:** the Allen CCF 2017 structure meshes (`.obj`, one per structure id)
  from the Allen Institute's download server. They are downloaded once, on
  first use, into `data_root/atlas/ccf_2017_meshes/`, never into the repo, and
  their sizes are listed before downloading.
- **three.js:** kept in the repo as a local file, with no CDN, so the app works
  offline. It is loaded as an ES module with no build step.

**2D probe strip:**
- The probe's channel map (`lateral_um`, `axial_um`, or `channel_positions.npy`
  for Phy), with each unit at its site, coloured by region at the chosen level.
- Region boundaries are marked along the shank. The selected unit is
  highlighted, and clicking a unit selects it.

**Phy data:** a Phy folder has no brain position, so the 3D view and region
filter are disabled with that reason. The probe strip still works from
`channel_positions.npy`. Importing channel locations (e.g. IBL's alignment
output) is a later step.

**Visual redesign:**
- **Colours:** one set of colour tokens, in light and dark, and region colours
  from Allen.
- **Layout:** a left rail (data source, QC toggle, level selector, region tree),
  the unit table, the plots, and a 3D/probe panel.
- **Plots:** restyled to match the rest of the app.
- **Readability:** trial counts and exclusions always visible.

**Chart library, a decision at the start of this step:** the PNG plots can't do
hover or zoom.
- **Recommended:** keep matplotlib, and make heatmap rows clickable through a
  row→unit map, with no chart library.
- **If hover and zoom on rasters are wanted:** uPlot, which is small, canvas
  based and MIT licensed. It's preferred over ECharts, which is large, and
  Plotly, which is excluded by CLAUDE.md §8.
- Whichever is chosen gets its own DECISIONS.md entry, like three.js.

**Tests first:**
- **Remapping:** Allen → Beryl → Cosmos checked against hand-picked acronyms
  (`CA1`→`CA1`→`HPF`, `DG-mo`→`DG`→`HPF`, `ml`→`root`).
- **Tree:** unit counts per node sum correctly over descendants.
- **Coordinates:** for d23a44ef, each unit's CCF position lies inside its
  region's voxels in the Allen annotation volume, for all units but a
  documented, small number. This needs iblatlas's annotation volume, a one-time
  download.
- **Track fit:** a straight probe gives a track through its sites.

## 3. Responsiveness against a shuffle null — built (2026-09-30)

Built (`docs/DECISIONS.md`, "Responsiveness against a shift null"), with one change
to the plan below: instead of 1,000 seeded random shifts, **every** circular shift
on a 5 ms grid is evaluated by FFT, so the test is deterministic and p is not
floored near 0.001. The refractory-period metric for Phy QC was added afterwards: a
port of ibllib's MIT `slidingRP_viol` (`docs/DECISIONS.md`, "Phy QC gains IBL's sliding
refractory-period test").


**What:** per unit and event, a p-value for "the rate in a response window
differs from the rate in a baseline window".

**Statistic:** mean over trials of (response rate − baseline rate).

**Null:** circularly shift each trial's event time by a random offset, keeping
trial structure and the spike train's autocorrelation. This is R4's
`null_shuffle` spirit, not the evaluation module's code.
- n shifts and the minimum shift go in `configs/analysis.yaml`.
- The seed is explicit (R7).
- p = (1 + #null ≥ observed) / (1 + n).

**Correction:** Benjamini–Hochberg across all units tested in one call, with the
number of tests shown next to every "responsive" label.

**Cross-validated heatmap sorting:** find peak times on odd trials and display
even trials. This fixes the prototype's known circularity.

**Tests first:**
- **Hand-computed case:** a unit that fires exactly 1 spike after every event
  has p = 1/(n+1).
- **Calibration:** Poisson spike trains with no event locking. p is uniform (KS
  test), and BH's false discovery rate stays at or below α over many seeds. This
  is test input, never shown as data.
- **Seed:** a fixed seed reproduces p exactly.

**UI:** a "responsive" column in the unit table, with p and q values. The
population heatmap gains a "responsive only" filter.

**Phy QC:** a refractory-period metric was added (IBL's sliding RP test, MIT port).
Phy QC now passes 161 units on d23a44ef probe00, down from 200; IBL's label passes
114, and 101 pass both.

## 4. Project file and figure export — built (2026-09-30)

Built as planned (`docs/DECISIONS.md`, "Project files and figure export"):
- Opening, saving and exporting use command-line paths (`--project`) rather than a
  folder picker.
- Export writes SVG, PDF, JSON sidecars and a manifest to `runs/<time>_studio/`.


**Project file:** `*.ndstudio.json`, a plain JSON file that holds:
- the data source (backend and eid, or Phy path), with a hash of the spike files;
- the QC config hash;
- the selected units and filters, including the region level and tree
  selection;
- each analysis's parameters: event, window, bin, baseline, seed.

It holds no results. Reopening it recomputes everything, so a project can never
show numbers that disagree with the data. If the hashes changed, it warns and
names the file.

**Export:**
- Each figure as SVG and PDF, from the same `viz/` functions at a set size and
  font.
- A sidecar `.json` next to each figure, with every number plotted: bin centres,
  mean, SEM, n_trials, n_excluded, unit ids, and the parameters. Anything a
  figure shows is then traceable to a file (§6's spirit).
- Figures and sidecars go to `runs/<run_id>/` with the manifest §7 already asks
  for: git SHA, config hash and seed.

**Tests:**
- Round trip: save a project, reload it, and get identical PSTH arrays.
- A changed spike file triggers the hash warning.
- The exported sidecar equals the engine's output.

## Further steps: 4b and 5–14 (each marked planned or built)

Each step says:
- what it adds;
- the null it needs if it labels units (R4: a label such as "tuned" or
  "connected" is a claim, and ships with a null and a multiple-testing correction
  over the units tested);
- what Phy data lacks for it;
- its tests.

As in steps 1–4, the engine goes in `analysis/`, the UI draws only, and a test is
written first where the contract is clear.

## 4b. Homepage and data selection — part (a) built, part (b) planned (2026-09-30)

Added before step 5, as **4b** so the later steps keep their numbers.

**Part (a) as built** (`docs/DECISIONS.md`, "Homepage and data selection, part
(a)"), with these differences from the plan below:
- The manifest went to version 2, with `region_units` plus motion-energy and pupil
  modalities; it is stored in `derived/manifest-v2/`.
- The default trial filter is `bwm_include` plus excluding no-go, set in
  `configs/catalog.yaml`. It also applies to sessions opened with `--eid` or
  `--phy`.

**Part (a): the homepage, filters, trial filters and opening a session.**
- **Starting up:** the server starts with no session and serves a homepage at
  `/`. Opening a session is a POST behind the existing Origin and Content-Type
  checks. It loads through `load_session` with a visible loading state, then
  shows the session view, with a Home link back. `--eid`, `--phy` and
  `--project` still skip the homepage.
- **Switching session** resets the unit selection, caches and test results.
- **Session list:** built from the BWM manifest (`data/manifest.py`), written once
  under `derived/` if no copy exists. Filtering and counts live in
  `analysis/catalog.py`, not the page. It shows a sortable table (lab, subject,
  date, probes, units, trials, included trials, Beryl regions) and marks
  sessions already in the local cache.
- **Session filters,** with live counts ("N sessions, M probes, K units match"):
  - lab, subject and date range;
  - a region from the iblatlas tree, with descendants and a minimum number of
    units in configs;
  - minimum good units, minimum included trials, and number of probes;
  - behaviour modalities.
- **Unit counts are labelled** as the release's good units, not Studio's QC
  count. For d23a44ef, Studio QC passes 390 of the release's 398.
- **Trial filters** (per trial: every BWM session runs the same task):
  - `bwm_include`, contrasts, block, outcome, and excluding no-go trials.
  - They are built on step 5's `analysis/conditions.py` definitions, so there is
    one definition only.
  - Excluded trials are counted, and every caption gives the trial count after
    filtering.
  - The filters are part of the responsiveness and selectivity cache keys, the
    project file and the export records.
  - Phy sessions offer only what their events CSV has, with the reason shown for
    the rest.
- **Tests:**
  - each session filter on a hand-built manifest;
  - region counts with descendants;
  - trial-filter counts, with excluded trials counted;
  - a test result is never shown under another trial filter;
  - a project round trip with trial filters;
  - no session at start, old flags skip the homepage, and switching resets state.

**Part (b): 3D overview, session sets, Phy folders, recent projects.**
- **3D overview:** every matching probe drawn from the manifest's tip and top
  positions, converted to CCF by `analysis/atlas.py`, coloured by lab. Hovering
  names the session and probe; clicking opens it. It updates with the filters.
- **Session sets:** named JSON files holding eids, the manifest version, the
  filters used and a hash. Reopening one warns if the manifest version changed.
  Step 12 will use them. The session view still holds one session at a time.
- **"Open a Phy folder":** a path field completing against a configured Phy root,
  pairing each folder with the `events.csv` beside its `params.py`. It refuses
  paths outside the root, traversal and symlinks out, and missing files are
  refused in plain language.
- **Recent projects** from `data_root/projects`, newest first.
- **Tests:**
  - probe lines converted exactly as unit positions are;
  - a session-set round trip, and the warning on a changed manifest version;
  - Phy path refusals;
  - the recent-projects list matches the folder.

## 5. Condition-split PSTHs and tuning — built (2026-09-30)

Built (`docs/DECISIONS.md`, "Condition-split PSTHs, tuning curves and
selectivity").
- **As planned:** choice is permuted within signed-contrast strata, and block
  uses pseudo-sessions.
- **Also stratified:** side within choice, and outcome within signed contrast.
- **Strata:** 0% contrast is split by side.
- **Block window:** block uses the pre-event window.
- **Validity:** the pseudo-session null is valid on average over block
  sequences, not for every single session.


**Adds:**
- **Condition-split PSTHs:** PSTHs split by a task variable (stimulus side, signed
  contrast, choice, feedback, block prior), overlaid, with a trial count per
  condition.
- **Tuning curves:** response-window rate against signed contrast, mean ± SEM
  per level.
- **Where:** `analysis/conditions.py` and `analysis/tuning.py`.

**Null, for "selective" or "tuned" labels:**
- **Why a plain shuffle fails:** IBL's task variables are correlated. Choice
  follows stimulus side, and the block prior predicts side. So a plain label
  shuffle calls a purely stimulus-driven unit "choice-selective".
- **Choice and side:** a conditional permutation, shuffling labels only within
  strata of the other variables (the condition-combined test IBL's Brain Wide Map
  used).
- **Block:** the pseudo-session null (`evaluation/nulls.py` already has a seeded
  port of IBL's block generator), because blocks are autocorrelated.
- **Tuning:** Spearman correlation with contrast, against contrast labels
  permuted within stimulus side.
- **Correction:** Benjamini–Hochberg (BH) across the units tested, with explicit
  seeds.

**Phy data:**
- Every condition needs its canonical column in the events CSV (`contrastLeft`,
  `contrastRight`, `choice`, `feedbackType`, `probabilityLeft`). A condition
  without its column is unavailable, with the reason shown.
- The pseudo-session null exists only for IBL's block generator, so block
  selectivity is refused for other tasks.

**Tests:**
- A split PSTH checked by hand, extending the prototype's hand-computed PSTH
  example with condition labels.
- A tuning curve checked by hand.
- **Calibration:** a simulated unit driven only by stimulus side, with choice
  correlated to side:
  - the conditional null keeps choice p-values uniform;
  - a plain shuffle doesn't. This is the test that justifies the null.
- The pseudo-session null reused unchanged.

## 6. Movement controls

**Adds:** separates rate changes around an event from movement, the confound step 3
found. 320 of 390 units "change around stimulus onset", with the baseline in the
quiescence period.
- **Wheel speed:** the wheel-speed PSTH drawn beside the neural one, from
  `behaviour.wheel`.
- **Reaction time:** stimulus-aligned PSTHs split by early and late movers.
- **A movement-free test:** the responsiveness test restricted to trials where
  the first movement comes after the response window ends.
- **A "movement-locked" label:** it compares alignment to each trial's own first
  movement with alignment to shifted movement times.

**Null:**
- **Movement-free test:** step 3's circular-shift null on the restricted trials,
  reporting how many trials remain.
- **"Movement-locked":** reaction times permuted across trials within each
  contrast level. A unit is labelled only if alignment to its own trial's movement
  beats the permuted alignments.
- **Correction:** BH across the units tested.

**Phy data:**
- Phy import reads spikes and events only, so there is no wheel. The wheel-speed
  panel and wheel-based movement onsets are declared missing.
- Only `firstMovement_times` supplied in the events CSV enables the movement-free
  test.

**Tests:**
- Hand-checked trial selection by reaction time.
- A hand-checked wheel-speed alignment.
- **Simulation:** a unit locked only to movement:
  - responsive at stimulus onset over all trials;
  - not responsive in movement-free trials;
  - labelled movement-locked.
- Null calibration on units with no locking.

## 7. Unit quality panel

**Adds:** a per-unit panel showing:
- the ISI histogram with refractory lines;
- the autocorrelogram;
- the sliding refractory-period details (pair counts against the Poisson bound
  at each refractory period tested);
- firing rate and spike count across the session (presence and stability);
- the mean waveform when available;
- every QC reason.

**Null:** none new. It shows the QC results already computed. The sliding RP test
already carries its own confidence level, and it labels quality, not responses. Any
new quality label (drift, say) must use IBL's metric definition and its threshold
from config.

**Phy data:**
- **Waveforms:** they need the raw `.dat` file (not read) or `templates.npy` and
  the whitening inverse.
- **Amplitudes:** `amplitudes.npy` is in template units, not volts, so IBL's
  amplitude and noise-cutoff metrics stay missing.
- **IBL waveforms:** available through ONE (`clusters.waveforms`, cached for
  d23a44ef).

**Tests:**
- ISI histogram and autocorrelogram counts checked by hand.
- The panel's sliding-RP details reproduce `sliding_rp_pass`'s verdict for every
  probe00 cluster.
- A presence ratio checked by hand.

## 8. Cross-correlograms

**Adds:**
- **Views:** cross-correlograms for chosen pairs, within or across probes, raw
  and jitter-corrected.
- **Label:** putative monosynaptic connections.

**Null, for "connected":**
- **Test:** interval jitter (Amarasingham et al. 2012). Spikes are resampled
  within fixed windows of a few ms, which keeps slow co-modulation and breaks
  millisecond timing.
- **Correction:** BH across the pairs tested, reporting their number. Pairs grow
  as n².
- **Seed:** explicit.

**Phy data:**
- Works from spike times alone.
- Sorting hides near-simultaneous spikes on nearby channels (overlapping
  templates), which makes a false dip at zero lag. Close pairs need
  `channel_positions.npy` to be flagged; without it the flag is declared missing.

**Tests:**
- Cross-correlogram counts checked by hand.
- **Calibration:** jitter p-values are uniform on independent Poisson pairs.
- An injected 2 ms excitatory coupling is detected.
- A same-channel pair is flagged.

## 9. Population trajectories

**Adds:**
- **Plot:** condition-averaged population activity projected on principal
  components, in 2D or 3D.
- **Cross-validation:** components are fit on odd trials and data projected from
  even trials, the heatmap rule from step 3.
- **Naming:** axes are named `pc_k` (R5): no interpretation in code or plots.

**Null:** none while it is descriptive. A claim that trajectories differ between
conditions would need a distance statistic against condition labels permuted
within strata (as in step 5). Until that exists, no labels.

**Phy data:** works. There is no region grouping without channel locations (step 13).

**Tests:**
- Planted low-rank data is recovered.
- Cross-validated projection never uses held-out trials in the fit.
- The page shows no labels.

## 10. Decoding in Studio

**Adds:**
- **Scope:** decode a task variable from the selected units of one session
  through the existing six-row evaluation contract (`evaluation/contract.py`),
  with the split registry (R1, R2).
- **Display:** the page shows the full table and the null verdicts, never a bare
  score, per the Phase 8b rules.
- **Phase 3 result:** the gate did not pass (`docs/NEGATIVE_RESULTS.md`). So a
  result that doesn't beat `null_trialstruct` is said plainly.

**Null:** already in the contract: `null_shuffle`, `null_trialstruct`, and
pseudo-sessions for block.

**Phy data:**
- Targets need canonical trial columns, and wheel velocity needs a wheel (see
  step 6).
- The split registry is built around IBL session manifests. It needs a
  single-session path for a non-IBL folder, still with whole-trial blocks and a
  gap.

**Tests:**
- Studio's result equals the CLI's for the same units, split and seed.
- The guards refuse a random time-point split.
- The page renders all six rows and the verdict.

## 11. Unit browsing (session picking moved to step 4b)

**Adds:**
- **Opening sessions: replaced by step 4b** (homepage and data selection), which
  covers the session list, Phy folders and the folder-picker question.
- **Browsing units:** unit search by id or region, next and previous with the
  keyboard, and pinned units.

**Null:** none. There are no labels.

**Phy data:** nothing beyond step 4b.

**Tests:**
- Search by id and region.
- Keyboard next and previous follow the table's order.
- Pinned units survive a filter change.

## 12. Across-session region summaries

**Adds:**
- **Per region:** the fraction of responsive or tuned units per Beryl region
  across sessions.
- **Distribution:** always shown per session, never only pooled (§5).
- **Map:** a Swanson flatmap from iblatlas.

**Null, for region-level claims** (for example, "region X has more responsive
units than chance"):
- **Test:** sessions are the unit of inference. The null permutes region labels
  across units within each session, or uses a mixed model with session as a
  random effect.
- **Correction:** FDR across regions.
- **Minimum:** a minimum number of sessions per region, with smaller regions
  refused.
- **Counts:** units and sessions are reported per region.

**Phy data:** no regions, so excluded until channel locations are imported (step 13).

**Tests:**
- Aggregation checked by hand.
- The per-session distribution is present.
- A region below the session minimum is refused.
- **Calibration:** the region-level null stays calibrated when units are pooled
  unevenly across sessions.

## 13. Phy sync and channel locations

**Adds:**
- **Sync:** event times are aligned from the behaviour or NIDQ clock to the
  probe clock, from SpikeGLX sync pulses (offset plus linear drift). The fit
  residuals are reported, and it refuses above a tolerance. This replaces step
  1's rule that events must already be on the probe clock.
- **Channel locations:** histology-aligned channel positions are imported (IBL's
  alignment output `channel_locations.json`, or a CSV of channel to CCF position
  and acronym). This gives Phy units regions, the region tree and the 3D view.

**Null:** none. Validation is by fit residuals and known answers, not labels.

**Phy data:** this step fills the gaps listed in steps 1, 2, 5 and 12.

**Tests:**
- **Sync:**
  - planted drift and offset are recovered from simulated pulses;
  - on d23a44ef, sync pulses from ONE reproduce IBL's own alignment within a
    stated tolerance.
- **Channel locations:** an import round trip, with units mapping to the same
  acronyms as the BWM backend.

## 14. Installer (Phase 8b)

**Adds:**
- **Installer:** a one-step installer for macOS, Windows and Linux. The options
  to evaluate are conda constructor, pixi and PyInstaller.
- **Contents:** Python 3.11, the dependencies with their pins (the llvmlite/numba
  note in DECISIONS), and the vendored three.js.
- **In-app data:** downloads with progress for sessions, meshes and volumes.
- **Licence first:** the repo needs a licence before anything is distributed.
  This is why the GPL `slidingRP` package was not used.

**Null:** none.

**Phy data:** nothing specific, but folders must stay local and are never
uploaded.

**Tests:**
- A clean-machine install smoke test per OS in CI.
- The app starts offline with its vendored assets.
- A check that every bundled dependency's licence is compatible with the repo's.

## Not in these steps

- Spike sorting and curation. They stay upstream by design (CLAUDE.md §2).
- Hosted deployment. Studio is a local app only.
