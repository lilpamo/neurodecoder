# Decisions

Record every non-obvious decision here: added dependencies, deviations from
CLAUDE.md, and choices about scope from §2. One entry per decision, newest
first.

---

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

**Decision:** Patched one line in `external/NEDS/src/utils/ibl_data_utils.py`
(`SessionLoader(one, eid=eid)` → `SessionLoader(one=one, eid=eid)`) to keep the
Phase 0 reproduction attempt moving. This is a local patch to a gitignored
external clone, not a change to our own code.

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
