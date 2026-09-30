# Neurodecoder Studio: the next three steps

Proposal, 2026-09-30. Follows the prototype on `studio-prototype`
(`docs/DECISIONS.md`, "Direction change: Neurodecoder Studio"). Each step keeps
the prototype's rule: the UI calls `neurodecoder/analysis/` (or `data/`) only.

## 1. Import Kilosort / Phy folders (`data/backends/phy.py`), about 1–2 sessions

**What:** a fourth backend that returns the same `Session` as `bwm`, `nwb` and
`one`, so everything downstream works unchanged.
- **Read:** `spike_times.npy` (samples), `spike_clusters.npy`, `params.py`
  (`sample_rate`), `channel_positions.npy`, and the cluster table.
- **Cluster table:** `cluster_info.tsv` if Phy saved one, else
  `cluster_group.tsv` and `cluster_KSLabel.tsv`.
- **Dependencies:** none; NumPy and pandas are enough.
- **Depth:** the amplitude-weighted channel y position.
- **Labels:** Phy's `good`/`mua`/`noise` are kept as a new `curation` column.
  They are never mapped onto IBL's numeric `label`.

**Events:** a CSV of event times, one column per event, one row per trial. The
first version **requires the times to already be on the probe's clock, in
seconds**. Otherwise it refuses, and the refusal says so. Sync (NIDQ or
SpikeGLX pulse alignment) is a separate step.

**Missing means missing:**
- A Phy folder has no brain region, so `units.acronym` is declared missing and the
  region filter is disabled with a reason.
- Unit QC today needs a label and a location. For Phy data the options are:
  - a Phy-specific QC config (curation == good + task rate);
  - failing every unit. **Needs your decision.**

**Tests first:**
- A tiny Phy folder written in the test from known arrays, checked against a
  hand-computed spike train per cluster.
- A refusal test for events off the recording's clock.
- Samples-to-seconds checked at 30 kHz.

**Open question:** does "import" mean a folder picker in the UI? That needs a
local file dialog: a path text box, or a native dialog via the OS.

## 2. Responsiveness against a shuffle null (`analysis/responsiveness.py`), about 1–2 sessions

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

## 3. Project file and figure export (`studio/project.py`), about 1 session

**Project file:** `*.ndstudio.json`, a plain JSON file that holds:
- the data source (backend and eid, or Phy path), with a hash of the spike files;
- the QC config hash;
- the selected units and filters;
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

## Not in these steps

- Tuning curves (contrast, choice).
- Sync or alignment tools.
- Multiple sessions per project.
- The installer (Phase 8b).
- An interactive zoom library.
