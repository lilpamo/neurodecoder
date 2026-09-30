"""Neurodecoder Studio prototype: a local web UI over one sorted session.

Standard library HTTP server, no new dependency. It loads the session once through
data.load.load_session and answers each request by calling analysis/ and viz/ only.

    python -m neurodecoder.studio.server --eid d23a44ef-1402-4ed7-97f5-47e9a7a504d9
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from neurodecoder.analysis.events import EVENTS, event_times
from neurodecoder.analysis.psth import (
    bin_edges,
    peak_order,
    population_psth,
    psth,
    raster,
    scale_rows_for_display,
    selection_average,
)
from neurodecoder.analysis.units import unit_table
from neurodecoder.data.load import load_session
from neurodecoder.qc.units import load_qc_config
from neurodecoder.viz.studio_plots import population_figure, unit_figure

PAGE = Path(__file__).with_name("index.html")


class Studio:
    def __init__(self, eid: str, backend: str):
        self.session = load_session(eid, backend)
        self.units = unit_table(self.session, load_qc_config())

    def _select(self, q: dict) -> "list[str]":
        rows = self.units if q.get("all") == "1" else self.units[self.units["qc_passed"]]
        if q.get("region"):
            rows = rows[rows["region"] == q["region"]]
        return list(rows.index)

    def _params(self, q: dict):
        window = (float(q["t0"]), float(q["t1"]))
        baseline = (float(q["b0"]), float(q["b1"])) if q.get("baseline") == "1" else None
        events = event_times(self.session.trials, q["event"])
        return window, float(q["bin"]), baseline, events

    def units_json(self, q: dict) -> dict:
        shown = self.units.loc[self._select({**q, "region": ""})]
        rows = shown.reset_index().rename(columns={"unit_id": "id"})
        return {
            "eid": self.session.eid,
            "n_trials": self.session.n_trials,
            "n_units_total": len(self.units),
            "n_units_passing": int(self.units["qc_passed"].sum()),
            "events": {k: v[0] for k, v in EVENTS.items()},
            "regions": sorted(shown["region"].unique()),
            "units": json.loads(rows.to_json(orient="records")),
        }

    def unit_png(self, q: dict) -> bytes:
        window, bin_width, baseline, events = self._params(q)
        spikes = self.session.spikes[q["unit"]]
        p = psth(spikes, events, window, bin_width, baseline)
        trial, rel = raster(spikes, events, window)
        region = self.units.at[q["unit"], "region"]
        title = f"{q['unit']} ({region}) · {EVENTS[q['event']][0]} · n = {p.n_trials} trials" + (
            f", {p.n_excluded} without this event excluded" if p.n_excluded else ""
        )
        return unit_figure(trial, rel, p, window, title, baseline is not None)

    def population_png(self, q: dict) -> bytes:
        window, bin_width, baseline, events = self._params(q)
        ids = self._select(q)
        if not ids:
            raise ValueError("no units match this filter")
        pop = population_psth(self.session.spikes, ids, events, window, bin_width, baseline)
        order = peak_order(pop)
        mean, sem = selection_average(pop)
        edges = bin_edges(window, bin_width)
        title = (
            f"{len(ids)} units ({q.get('region') or 'all regions'}) · "
            f"{EVENTS[q['event']][0]} · n = {int(np.isfinite(events).sum())} trials"
        )
        return population_figure(
            scale_rows_for_display(pop[order]),
            (edges[:-1] + edges[1:]) / 2,
            mean,
            sem,
            window,
            title,
        )


def make_handler(studio: Studio):
    routes = {
        "/api/units": ("application/json", lambda q: json.dumps(studio.units_json(q)).encode()),
        "/api/unit.png": ("image/png", studio.unit_png),
        "/api/population.png": ("image/png", studio.population_png),
        "/": ("text/html; charset=utf-8", lambda q: PAGE.read_bytes()),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path not in routes:
                return self._send(404, "text/plain", b"not found")
            ctype, fn = routes[url.path]
            try:
                self._send(200, ctype, fn(q))
            except (ValueError, KeyError) as e:
                # Plain-language refusal for the page to show, never a traceback.
                self._send(400, "text/plain; charset=utf-8", f"Cannot plot this: {e}".encode())

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eid", default="d23a44ef-1402-4ed7-97f5-47e9a7a504d9")
    ap.add_argument("--backend", default="bwm")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    studio = Studio(args.eid, args.backend)
    print(f"Neurodecoder Studio: http://127.0.0.1:{args.port}  ({args.eid})", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(studio)).serve_forever()


if __name__ == "__main__":
    main()
