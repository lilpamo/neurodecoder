# Decisions

Record every non-obvious decision here: added dependencies, deviations from
CLAUDE.md, and choices about scope from §2. One entry per decision, newest
first.

---

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
