# Decisions

Record every non-obvious decision here: added dependencies, deviations from
CLAUDE.md, and choices about scope from §2. One entry per decision, newest
first.

---

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
