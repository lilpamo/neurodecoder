# Decisions

Record every non-obvious decision here: added dependencies, deviations from
CLAUDE.md, and choices about scope from §2. One entry per decision, newest
first.

---

### 2026-09-27 — DANDI NWB backend mappings (`neurodecoder/data/backends/dandi_nwb.py`)

**Decision:** `load_session_nwb(path)` reads DANDI 000409 `desc-processed` NWB
files into a `Session`. For now it reads local files; streaming via `remfile`
comes next. Every mapping was checked value by value against ONE for session
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
unit ID is (`cluster_uuid` vs `(pid, cluster_id)`) is left to each backend
and will be settled when the first two backends are compared. BWM stores
trial times as float32 (about 0.5 ms resolution on an hour-long session), so
cross-backend comparisons of trial times need a tolerance of about 1 ms,
while counts stay exact.

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
