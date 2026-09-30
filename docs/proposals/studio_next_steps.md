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
- **Phy QC:** group in `[good]` and a task-period rate of at least 0.1 Hz.
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
floored near 0.001. The refractory-period metric for Phy QC is still open.


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

**Phy QC:** consider adding a refractory-period metric computed from spike
times. Kilosort's `good` passes 200 units on d23a44ef probe00, where IBL's label
passes 114 (see step 1's DECISIONS entry).

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

## Not in these steps

- Tuning curves (contrast, choice).
- Sync or alignment tools, and importing channel locations for Phy data.
- Multiple sessions per project.
- The installer (Phase 8b).
