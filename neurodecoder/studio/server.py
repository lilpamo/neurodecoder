"""Neurodecoder Studio: a local web UI over one sorted session.

Standard-library HTTP server, no web framework. It loads the session once and answers
each request by calling analysis/ (numbers), viz/ (PNGs) and data/atlas_meshes
(cached Allen meshes). The page draws what it is sent and computes nothing.

    python -m neurodecoder.studio.server --eid d23a44ef-1402-4ed7-97f5-47e9a7a504d9
    python -m neurodecoder.studio.server --phy FOLDER --events events.csv
"""

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd

from neurodecoder.analysis.atlas import (
    LEVELS,
    ROOT,
    ccf_um,
    depth_runs,
    probe_track,
    region_at_level,
    region_info,
    region_tree,
    units_in_node,
)
from neurodecoder.analysis.events import EVENTS, available_events, event_times
from neurodecoder.analysis.psth import (
    alternate_halves,
    bin_edges,
    peak_order,
    population_psth,
    psth,
    raster,
    scale_rows_for_display,
    selection_average,
)
from neurodecoder.analysis.responsiveness import load_response_config, responsiveness
from neurodecoder.analysis.units import unit_table
from neurodecoder.data.atlas_meshes import mesh_path
from neurodecoder.data.load import load_data_config
from neurodecoder.studio.export import export_view
from neurodecoder.studio.project import (
    DEFAULT_VIEW,
    SUFFIX,
    Source,
    load_source,
    make_project,
    open_project,
    save_project,
)
from neurodecoder.viz.studio_plots import THEMES, population_figure, probe_colours, unit_figure

HERE = Path(__file__).parent
STATIC = HERE / "static"
DEFAULT_LEVEL = "Beryl"
BRAIN_ID = 997  # Allen structure id of the whole brain ("root")
DEFAULT_EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
_MESH = re.compile(r"^/mesh/(\d+)\.obj$")
_MAX_BODY = 1 << 20
RUNS = Path(__file__).resolve().parents[2] / "runs"


def _records(frame: pd.DataFrame) -> list[dict]:
    """JSON-ready rows; NaN becomes null."""
    return json.loads(frame.to_json(orient="records"))


class Studio:
    def __init__(
        self,
        session,
        qc,
        atlas_root: Path,
        source: Source | None = None,
        project_path: Path | None = None,
        view: dict | None = None,
        warnings: list[str] = (),
    ):
        self.session = session
        self.qc = qc
        self.source = source  # where the session came from; recorded in projects and exports
        self.project_path = project_path
        self.view = {**DEFAULT_VIEW, **(view or {})}
        self.warnings = list(warnings)
        self.units = unit_table(session, qc)
        self.atlas_root = atlas_root
        self.has_regions = "units.acronym" in session.available.present
        self.has_positions = all(f"units.{a}" in session.available.present for a in "xyz")
        self.response_cfg = load_response_config()
        self.probes = sorted(self.units["probe"].unique())
        # (event, all units, probe) -> result: BH ran over exactly that unit set.
        self._tests: dict[tuple[str, bool, str], pd.DataFrame] = {}

    def _test_key(self, q: dict) -> tuple[str, bool, str]:
        return q.get("event", ""), q.get("all") == "1", q.get("probe", "")

    def _tested(self, q: dict) -> pd.DataFrame | None:
        return self._tests.get(self._test_key(q))

    def _level(self, q: dict) -> str:
        level = q.get("level", DEFAULT_LEVEL)
        if level not in LEVELS:
            raise ValueError(f"unknown region level {level!r}")
        return level

    def _regions(self, q: dict) -> pd.Series:
        """(n_units,) region at the requested level; None without regions."""
        if not self.has_regions:
            return pd.Series(None, self.units.index, dtype=object)
        values = region_at_level(self.units["region"].to_numpy(), self._level(q))
        return pd.Series(values, self.units.index, dtype=object)

    def _select(self, q: dict) -> list[str]:
        keep = np.ones(len(self.units), bool) if q.get("all") == "1" else self.units["qc_passed"]
        if q.get("probe"):
            if q["probe"] not in self.probes:
                raise ValueError(f"no probe {q['probe']!r} in this session; it has {self.probes}")
            keep = keep & (self.units["probe"] == q["probe"])
        if q.get("node"):
            keep = keep & units_in_node(self._regions(q).to_numpy(), q["node"])
        if q.get("responsive") == "1":
            tested = self._tested(q)
            if tested is None:
                raise ValueError("run the responsiveness test for this event first")
            keep = keep & self.units.index.isin(tested.index[tested["responsive"]])
        return list(self.units.index[np.asarray(keep, bool)])

    def _params(self, q: dict):
        window = (float(q["t0"]), float(q["t1"]))
        baseline = (float(q["b0"]), float(q["b1"])) if q.get("baseline") == "1" else None
        events = event_times(self.session.trials, q["event"])
        return window, float(q["bin"]), baseline, events

    def session_json(self, q: dict) -> dict:
        missing = self.session.available.missing
        return {
            "eid": self.session.eid,
            "n_trials": self.session.n_trials,
            "n_units_total": len(self.units),
            "n_units_passing": int(self.units["qc_passed"].sum()),
            "events": available_events(self.session.trials),
            "levels": list(LEVELS),
            "default_level": DEFAULT_LEVEL,
            "missing": {k: v for k, v in missing.items() if k.startswith("units.")},
            "response": dict(self.response_cfg.__dict__),
            "probes": self.probes,
            "probe_colours": {theme: probe_colours(self.probes, theme) for theme in THEMES},
            "project": {
                "path": None if self.project_path is None else str(self.project_path),
                "view": self.view,
                "warnings": self.warnings,
            },
        }

    def save(self, view: dict) -> dict:
        """Write the project file: source, hashes, configs and this view, never results."""
        if self.source is None or self.project_path is None:
            raise ValueError("this Studio was started without a data source to save")
        save_project(make_project(self.source, self.session, self.qc, view), self.project_path)
        self.view = {**DEFAULT_VIEW, **view}
        return {"path": str(self.project_path)}

    def export(self, view: dict) -> dict:
        out = export_view(self, view, RUNS)
        return {"folder": str(out), "files": sorted(p.name for p in out.iterdir())}

    def test_json(self, q: dict) -> dict:
        """Run (or reuse) the responsiveness test on every shown unit for one event."""
        key = self._test_key(q)
        if key not in self._tests:
            ids = self._select({"all": q.get("all", "0"), "probe": q.get("probe", "")})
            if not ids:
                raise ValueError("no units to test")
            events = event_times(self.session.trials, q["event"])
            self._tests[key] = responsiveness(self.session.spikes, ids, events, self.response_cfg)
        return self._summary(self._tests[key])

    def _summary(self, t: pd.DataFrame) -> dict:
        up = t["responsive"] & (t["statistic_hz"] > 0)
        return {
            "n_tests": int(t["n_tests"].iloc[0]),
            "n_responsive": int(t["responsive"].sum()),
            "n_up": int(up.sum()),
            "n_down": int((t["responsive"] & ~up).sum()),
            "n_shifts": int(t["n_shifts"].iloc[0]),
            "n_trials": int(t["n_trials"].iloc[0]),
            "n_excluded": int(t["n_excluded"].iloc[0]),
            "probes": sorted(self.units.loc[t.index, "probe"].unique()),
        }

    def units_json(self, q: dict) -> dict:
        """The units in the selected tree node; the tree counts every unit shown."""
        regions = self._regions(q)
        everywhere = self._select({**q, "node": ""})
        shown = self.units.loc[self._select(q)]
        info = region_info(regions.dropna().unique()) if self.has_regions else {}
        rows = shown.assign(
            region_level=regions[shown.index],
            colour=[info.get(r, {}).get("colour") for r in regions[shown.index]],
        )
        tree = region_tree(regions[everywhere].to_numpy()) if self.has_regions else []
        tested = self._tested(q)
        if tested is not None:
            t = tested.reindex(rows.index)  # every shown unit was tested together
            tested_here = t["responsive"].notna()
            responsive = t["responsive"].where(tested_here, False).astype(bool)
            verdict = np.select(
                [~tested_here, ~responsive, t["statistic_hz"] > 0], [None, "no", "up"], "down"
            )
            rows = rows.assign(
                resp_q=t["q"], resp_p=t["p"], resp_hz=t["statistic_hz"], resp=verdict
            )
        return {
            "level": self._level(q) if self.has_regions else None,
            "units": _records(rows.reset_index().rename(columns={"unit_id": "id"})),
            "tree": tree,
            "test": None if tested is None else self._summary(tested),
        }

    def probe_json(self, q: dict) -> dict:
        """One probe's units and region runs along the shank, for the probe strip."""
        probe = self.units.at[q["unit"], "probe"]
        on_probe = self.units.loc[self._select({**q, "node": ""})]
        on_probe = on_probe[on_probe["probe"] == probe]
        regions = self._regions(q)[on_probe.index]
        info = region_info(regions.dropna().unique()) if self.has_regions else {}
        runs = depth_runs(on_probe["depth_um"].to_numpy(), regions.to_numpy())
        for run in runs:
            run["colour"] = info[run["region"]]["colour"]
        units = on_probe[["depth_um", "lateral_um"]].assign(
            region=regions, colour=[info.get(r, {}).get("colour") for r in regions]
        )
        return {
            "probe": probe,
            "runs": runs,
            "units": _records(units.reset_index().rename(columns={"unit_id": "id"})),
        }

    def geometry_json(self, q: dict) -> dict:
        """CCF positions (ap, dv, ml µm) of units and probe tracks, and meshes to draw."""
        if not self.has_positions:
            return {"missing": self.session.available.missing["units.x"]}
        shown = self.units.loc[self._select({**q, "node": ""})]
        ccf = ccf_um(self.session.units.loc[shown.index, ["x", "y", "z"]].to_numpy())
        regions = self._regions(q)[shown.index]
        info = region_info(regions.dropna().unique())
        tracks = []
        for probe, rows in shown.groupby("probe"):
            if len(rows) >= 2:
                a, b = probe_track(ccf[shown.index.get_indexer(rows.index)])
                tracks.append({"probe": probe, "a": a.tolist(), "b": b.tolist()})
        meshes = [{"acronym": a, **info[a]} for a in sorted(info) if a != ROOT]
        return {
            "brain_id": BRAIN_ID,
            "units": [
                {"id": u, "ccf": p.tolist(), "colour": info.get(r, {}).get("colour")}
                for u, p, r in zip(shown.index, ccf, regions)
            ],
            "tracks": tracks,
            "meshes": meshes,
        }

    def unit_data(self, q: dict) -> dict:
        """Every number the selected unit's figure plots, and its caption."""
        window, bin_width, baseline, events = self._params(q)
        spikes = self.session.spikes[q["unit"]]
        p = psth(spikes, events, window, bin_width, baseline)
        trial, rel = raster(spikes, events, window)
        region = self._regions(q)[q["unit"]]
        where = "" if pd.isna(region) else f" · {region}"
        caption = f"{q['unit']}{where} · {EVENTS[q['event']][0]} · n = {p.n_trials} trials" + (
            f", {p.n_excluded} without this event excluded" if p.n_excluded else ""
        )
        return {
            "trial": trial,
            "rel": rel,
            "psth": p,
            "window": window,
            "baseline": baseline,
            "caption": caption,
        }

    def unit_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.unit_data(q)
        theme = q.get("theme", "light")
        png = unit_figure(
            d["trial"], d["rel"], d["psth"], d["window"], d["baseline"] is not None, theme
        )
        return png, {"X-Caption": quote(d["caption"])}

    def population_data(self, q: dict) -> dict:
        """Every number the population figure plots, rows in plotted order, and its caption."""
        window, bin_width, baseline, events = self._params(q)
        ids = self._select(q)
        if not ids:
            raise ValueError("no units match this filter")
        # Sort on odd trials and show even ones, so the order is not fitted to what it shows.
        sort_on, show = alternate_halves(events)
        if show.size == 0:
            raise ValueError("needs at least 2 trials with this event: one half sorts, one shows")
        spikes = self.session.spikes
        order = peak_order(population_psth(spikes, ids, sort_on, window, bin_width, baseline))
        pop = population_psth(spikes, ids, show, window, bin_width, baseline)[order]
        mean, sem = selection_average(pop)
        edges = bin_edges(window, bin_width)
        which = "responsive " if q.get("responsive") == "1" else ""
        rows = [ids[i] for i in order]
        probes = self.units.loc[rows, "probe"].tolist()
        included = sorted(set(probes))
        caption = (
            f"{len(ids)} {which}units ({q.get('node') or 'all regions'}) · "
            f"{'probe' if len(included) == 1 else 'probes'} {', '.join(included)} · "
            f"{EVENTS[q['event']][0]} · sorted by peak time on odd trials (n = {sort_on.size}), "
            f"showing even trials (n = {show.size})"
        )
        return {
            "units": rows,
            "probes": probes,
            "rates_hz": pop,
            "scaled": scale_rows_for_display(pop),
            "bin_centers": (edges[:-1] + edges[1:]) / 2,
            "mean": mean,
            "sem": sem,
            "window": window,
            "baseline": baseline,
            "n_sort_trials": int(sort_on.size),
            "n_show_trials": int(show.size),
            "caption": caption,
        }

    def population_png(self, q: dict) -> tuple[bytes, dict]:
        d = self.population_data(q)
        theme = q.get("theme", "light")
        png, box = population_figure(
            d["scaled"],
            d["bin_centers"],
            d["mean"],
            d["sem"],
            d["window"],
            theme,
            row_groups=d["probes"],
            group_colours=probe_colours(self.probes, theme),
        )
        headers = {
            "X-Caption": quote(d["caption"]),
            "X-Rows": ",".join(d["units"]),
            "X-Box": ",".join(f"{v:.4f}" for v in box),
        }
        return png, headers

    def mesh(self, structure_id: int) -> bytes:
        return mesh_path(structure_id, self.atlas_root).read_bytes()


def _static(path: str) -> Path | None:
    """A file under static/, or None; never a path outside it."""
    target = (STATIC / path.removeprefix("/static/")).resolve()
    return target if target.is_file() and STATIC.resolve() in target.parents else None


def make_handler(studio: Studio):
    def as_json(fn):
        return lambda q: (json.dumps(fn(q)).encode(), {})

    routes = {
        "/api/session": ("application/json", as_json(studio.session_json)),
        "/api/units": ("application/json", as_json(studio.units_json)),
        "/api/probe": ("application/json", as_json(studio.probe_json)),
        "/api/geometry": ("application/json", as_json(studio.geometry_json)),
        "/api/test": ("application/json", as_json(studio.test_json)),
        "/api/unit.png": ("image/png", studio.unit_png),
        "/api/population.png": ("image/png", studio.population_png),
        "/": ("text/html; charset=utf-8", lambda q: ((HERE / "index.html").read_bytes(), {})),
    }
    posts = {"/api/project": studio.save, "/api/export": studio.export}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in routes:
                    ctype, fn = routes[url.path]
                    body, headers = fn(q)
                    return self._send(200, ctype, body, headers)
                if match := _MESH.match(url.path):
                    return self._send(200, "text/plain", studio.mesh(int(match.group(1))))
                if url.path.startswith("/static/") and (path := _static(url.path)):
                    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    return self._send(200, ctype, path.read_bytes())
                self._send(404, "text/plain", b"not found")
            except (ValueError, KeyError, OSError) as e:
                # Plain-language refusal for the page to show, never a traceback.
                self._send(400, "text/plain; charset=utf-8", f"Cannot show this: {e}".encode())

        def do_POST(self):
            # Only this page may post: JSON bodies force a CORS preflight, which this
            # server never answers, and the Origin must be this server's own.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self._send(403, "text/plain", b"cross-origin request refused")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._send(415, "text/plain", b"expected application/json")
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY:
                return self._send(413, "text/plain", b"request too large")
            fn = posts.get(urlparse(self.path).path)
            if fn is None:
                return self._send(404, "text/plain", b"not found")
            try:
                view = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, "application/json", json.dumps(fn(view)).encode())
            except (ValueError, KeyError, OSError) as e:
                self._send(400, "text/plain; charset=utf-8", f"Cannot do this: {e}".encode())

        def _send(self, code, ctype, body, headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return Handler


def _new_project_path(root: Path, name: str) -> Path:
    """root/<name>.ndstudio.json, or <name>-2, -3, ... so no project is overwritten."""
    path, n = root / f"{name}{SUFFIX}", 2
    while path.exists():
        path, n = root / f"{name}-{n}{SUFFIX}", n + 1
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eid", help="an IBL session (default d23a44ef-1402-4ed7-97f5-47e9a7a504d9)")
    ap.add_argument("--backend", default="bwm")
    ap.add_argument("--phy", help="a Kilosort/Phy output folder (one probe); needs --events")
    ap.add_argument("--events", help="CSV of trial events, seconds on the probe's clock")
    ap.add_argument(
        "--project",
        type=Path,
        help=f"a *{SUFFIX} file: alone, opens it; with a data source, saves a new one there",
    )
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if bool(args.phy) != bool(args.events):
        ap.error("--phy and --events go together")
    if args.phy and args.eid:
        ap.error("choose one data source: --eid or --phy")
    data = load_data_config()
    view, warnings = None, []
    if args.project and not (args.phy or args.eid):
        if not args.project.exists():
            ap.error(f"{args.project} does not exist; give a data source to start a new project")
        project, session, qc, warnings = open_project(args.project)
        source = Source(**{k: v for k, v in project["source"].items() if k != "release"})
        view, path = project["view"], args.project
    else:
        if args.phy:
            source = Source(
                kind="phy",
                folder=str(Path(args.phy).resolve()),
                events=str(Path(args.events).resolve()),
            )
        else:
            source = Source(kind="ibl", eid=args.eid or DEFAULT_EID, backend=args.backend)
        if args.project and args.project.exists():
            ap.error(f"{args.project} exists; open it with --project alone, or choose a new name")
        name = Path(source.folder).name if source.kind == "phy" else source.eid[:8]
        path = args.project or _new_project_path(data.data_root / "projects", name)
        session, qc = load_source(source)
    studio = Studio(session, qc, data.data_root / "atlas", source, path, view, warnings)
    print(f"Neurodecoder Studio: http://127.0.0.1:{args.port}  ({session.eid})", flush=True)
    print(f"Project file: {path}", flush=True)
    for warning in warnings:
        print(f"Warning: {warning}", flush=True)
    # Threads, so a long responsiveness test doesn't hold up the plots.
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(studio)).serve_forever()


if __name__ == "__main__":
    main()
