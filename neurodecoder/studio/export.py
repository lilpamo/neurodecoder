"""Figure export: the current view as SVG and PDF, with every plotted number beside it.

Each export is a new `runs/<run_id>/` folder (run_id = UTC time + "_studio"):
- `unit.svg`, `unit.pdf`, `unit.json`: the selected unit's raster and PSTH;
- `population.svg`, `population.pdf`, `population.json`: the heatmap and mean;
- `responsiveness.csv`, when the test was run for this event;
- `manifest.json`: git SHA and dirty flag, the project snapshot (source, file
  hashes, data fingerprint, config hashes), the view and the software versions.

Figures come from the same viz/ builders as the page, in the light theme at a fixed
size, titled with the page's caption, with text kept editable. The JSON sidecars hold
the arrays the figures were drawn from, so any number in a figure traces to a file.
Nothing is random, so there is no seed.
"""

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neurodecoder.studio.project import REPO, make_project, view_to_query
from neurodecoder.viz.studio_plots import (
    build_population_figure,
    build_tuning_figure,
    build_unit_figure,
    condition_colours,
    probe_colours,
    save_vector,
)

THEME = "light"


def _jsonable(value):
    """Arrays become lists; NaN and infinities become null; numpy scalars become numbers."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return None if not math.isfinite(value) else float(value)
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=1, allow_nan=False))


def _git() -> dict:
    def git(*args):
        run = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
        return run.stdout.strip()

    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def export_view(studio, view: dict, runs_dir: str | os.PathLike) -> Path:
    """Write the view's figures, sidecars and manifest to a new run folder; return it."""
    q = view_to_query(view)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_studio"
    out = Path(runs_dir) / run_id
    out.mkdir(parents=True)
    files = []

    if view.get("unit"):
        d = studio.unit_data(q)
        fig = build_unit_figure(
            d["groups"], d["window"], d["baseline"] is not None, THEME, d["caption"]
        )
        for suffix in ("svg", "pdf"):
            save_vector(fig, out / f"unit.{suffix}")

        def trace(g):
            p = g.psth
            return {
                "condition": g.name,
                "bin_centers": p.bin_centers,
                "mean": p.mean,
                "sem": p.sem,
                "n_trials": p.n_trials,
                "n_excluded": p.n_excluded,
                "raster": {"trial": g.trial, "time_s": g.rel},
            }

        traces = [trace(g) for g in d["groups"]]
        sidecar = {
            "caption": d["caption"],
            "trials": studio._trial_summary(q),
            "unit": view["unit"],
            "split": view.get("split") or None,
            "window_s": d["window"],
            "baseline_s": d["baseline"],
        }
        if view.get("split"):
            sidecar["groups"] = traces
        else:
            sidecar["psth"] = {
                k: v for k, v in traces[0].items() if k not in ("condition", "raster")
            }
            sidecar["raster"] = traces[0]["raster"]
        _write_json(out / "unit.json", sidecar)
        files += ["unit.svg", "unit.pdf", "unit.json"]

        if view.get("split"):
            t = studio.tuning_data(q)
            curve = t["curve"]
            fig = build_tuning_figure(
                list(curve.index),
                curve["mean_hz"].to_numpy(),
                curve["sem_hz"].to_numpy(),
                curve["n"].tolist(),
                condition_colours(view["split"], t["levels"], THEME),
                view["split"] in ("contrast", "block"),
                THEME,
                t["caption"],
            )
            for suffix in ("svg", "pdf"):
                save_vector(fig, out / f"tuning.{suffix}")
            _write_json(
                out / "tuning.json",
                {
                    "caption": t["caption"],
                    "trials": studio._trial_summary(q),
                    "levels": curve.reset_index().to_dict(orient="list"),
                },
            )
            files += ["tuning.svg", "tuning.pdf", "tuning.json"]

    d = studio.population_data(q)
    fig, _ = build_population_figure(
        d["scaled"],
        d["bin_centers"],
        d["mean"],
        d["sem"],
        d["window"],
        THEME,
        d["caption"],
        row_groups=d["probes"],
        group_colours=probe_colours(studio.probes, THEME),
    )
    for suffix in ("svg", "pdf"):
        save_vector(fig, out / f"population.{suffix}")
    _write_json(out / "population.json", {**d, "trials": studio._trial_summary(q)})
    files += ["population.svg", "population.pdf", "population.json"]

    tested = studio._tested(q)
    if tested is not None:
        tested.to_csv(out / "responsiveness.csv")
        files.append("responsiveness.csv")
    chosen = studio._selectivity.get(studio._selectivity_key(q))
    if chosen is not None:
        chosen.to_csv(out / "selectivity.csv")
        files.append("selectivity.csv")

    manifest = {
        "run_id": run_id,
        "created": datetime.now(UTC).isoformat(),
        "command": "neurodecoder.studio export",
        "git": _git(),
        "seed": None,
        "project": make_project(studio.source, studio.session, studio.qc, view),
        "view": view,
        "responsiveness": None if tested is None else asdict(studio.response_cfg),
        "selectivity": None if chosen is None else asdict(studio.selectivity_cfg),
        "files": files,
        "versions": {
            "python": platform.python_version(),
            **{m: sys.modules[m].__version__ for m in ("numpy", "pandas", "scipy", "matplotlib")},
        },
    }
    _write_json(out / "manifest.json", manifest)
    return out
