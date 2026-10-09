"""Publish thermal detections into Lattice as Entities, through the trust gate.

Default transport is a mock: it validates every entity against the real SDK
types (so the field mapping is proven correct) and records what would go on the
wire, without needing Lattice credentials. Pass --live with LATTICE_* env vars
set to use the real client instead.

    python -m lattice.publish                     # mock, full-severity frames
    python -m lattice.publish --severity 0.0      # clean frames (all trusted)
    python -m lattice.publish --limit 40 --json out.jsonl
    python -m lattice.publish --live              # real client, needs creds

What it prints is a scorecard, not a success message: how many frames fell into
each regime, how many tracks were asserted and how many of those were real
people vs ghosts, and how many real people were suppressed on blind frames
because the detector could not be trusted. That suppression is the honest cost
of the gate, so it is reported, not hidden.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from anduril import Entity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lattice.entity_mapper import (  # noqa: E402
    Detection, Regime, detection_to_entity_kwargs, regime_for_reliability,
    sensor_health_entity_kwargs,
)

DEFAULT_RECORDS = os.path.join(os.path.dirname(HERE), "results_thermal", "records.csv")
DEFAULT_AGG = os.path.join(os.path.dirname(HERE), "results_thermal", "aggregate.csv")


class MockEntityPublisher:
    """Validates via the real Entity type, then records. No network, no creds."""

    def __init__(self):
        self.published: list[Entity] = []

    def publish_entity(self, **kwargs) -> Entity:
        entity = Entity(**kwargs)          # real pydantic validation of the field mapping
        self.published.append(entity)
        return entity

    def wire_records(self) -> list[dict]:
        return [e.model_dump(by_alias=True, exclude_none=True) for e in self.published]


class LatticeEntityPublisher:
    """Thin wrapper over the real client. Credential-gated, so this path is only
    exercised with LATTICE_BASE_URL + LATTICE_TOKEN (or client id/secret) set."""

    def __init__(self):
        from anduril import Lattice
        token = os.environ.get("LATTICE_TOKEN")
        base_url = os.environ.get("LATTICE_BASE_URL")
        if not token or not base_url:
            raise SystemExit("--live needs LATTICE_BASE_URL and LATTICE_TOKEN in the environment.")
        self.client = Lattice(base_url=base_url, token=token)

    def publish_entity(self, **kwargs) -> Entity:
        return self.client.entities.publish_entity(**kwargs)


def load_reliability(aggregate_csv: str) -> dict[tuple[str, float], float]:
    """(perturbation, severity) -> measured mAP@0.5, the gate's input."""
    agg = pd.read_csv(aggregate_csv)
    return {(p, float(s)): float(m)
            for p, s, m in zip(agg.perturbation, agg.severity, agg["map"])}


@dataclass
class Scorecard:
    severity: float
    frames_by_regime: Counter = field(default_factory=Counter)
    tracks_published: int = 0
    tracks_real: int = 0           # published AND is_tp == 1
    tracks_ghost: int = 0          # published AND is_tp == 0
    suppressed_total: int = 0
    suppressed_real: int = 0       # real people dropped on blind frames
    health_signals: Counter = field(default_factory=Counter)
    entities_total: int = 0

    def render(self) -> str:
        L = []
        L.append(f"Trust-gated publish over frames at severity {self.severity:g}")
        L.append(f"  entities published (mock-validated): {self.entities_total}")
        L.append("")
        L.append("  frames by detector regime (gate reads measured mAP at the condition):")
        for r in (Regime.TRUSTED, Regime.DEGRADED, Regime.BLIND):
            L.append(f"    {r.value:8s} {self.frames_by_regime.get(r.value, 0):5d}")
        L.append("")
        L.append("  tracks asserted into the picture (trusted + degraded frames only):")
        L.append(f"    published      {self.tracks_published:5d}")
        L.append(f"      real person  {self.tracks_real:5d}")
        L.append(f"      ghost        {self.tracks_ghost:5d}")
        L.append("")
        L.append("  blind frames: tracks suppressed, producer health flipped to FAIL instead:")
        L.append(f"    suppressed           {self.suppressed_total:5d}")
        L.append(f"      of them real people{self.suppressed_real:5d}   <- honest cost of the gate")
        L.append(f"    FAIL health signals  {self.health_signals.get('BLIND', 0):5d}")
        L.append("")
        L.append("  the point: on a blind frame the operator sees one FAIL signal, not a")
        L.append("  sparse scatter of untrustworthy blips that looks like a nearly-clear field.")
        return "\n".join(L)


def run(records_csv: str, aggregate_csv: str, publisher, severity: float,
        limit: Optional[int] = None, now: Optional[dt.datetime] = None) -> Scorecard:
    now = now or dt.datetime.now(dt.timezone.utc)
    reliability = load_reliability(aggregate_csv)
    df = pd.read_csv(records_csv)
    df = df[df.severity == severity]
    # the dead control is not a field condition; drop it from the publish stream
    df = df[df.perturbation != "background_marks"]

    sc = Scorecard(severity=severity)
    frames = list(df.groupby(["image_id", "perturbation"]))
    if limit:
        frames = frames[:limit]

    for (image_id, pert), g in frames:
        map_at = reliability.get((pert, severity), 100.0)
        regime = regime_for_reliability(map_at)
        sc.frames_by_regime[regime.value] += 1

        dets = [Detection(int(r.cls), float(r.score), image_id, pert, severity,
                          int(r.is_tp) if not pd.isna(r.is_tp) else None) for r in g.itertuples()]

        if regime is Regime.BLIND:
            n_real = sum(1 for d in dets if d.is_tp == 1)
            sc.suppressed_total += len(dets)
            sc.suppressed_real += n_real
            publisher.publish_entity(**sensor_health_entity_kwargs(
                image_id, pert, severity, regime, map_at, n_suppressed=len(dets), now=now))
            sc.health_signals[regime.value] += 1
            sc.entities_total += 1
            continue

        # trusted / degraded: assert the tracks, and report producer health
        publisher.publish_entity(**sensor_health_entity_kwargs(
            image_id, pert, severity, regime, map_at, now=now))
        sc.health_signals[regime.value] += 1
        sc.entities_total += 1
        for i, d in enumerate(dets):
            publisher.publish_entity(**detection_to_entity_kwargs(d, regime, i, now=now))
            sc.tracks_published += 1
            sc.entities_total += 1
            if d.is_tp == 1:
                sc.tracks_real += 1
            elif d.is_tp == 0:
                sc.tracks_ghost += 1

    return sc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=DEFAULT_RECORDS)
    ap.add_argument("--aggregate", default=DEFAULT_AGG)
    ap.add_argument("--severity", type=float, default=1.0, help="frame severity to publish (default 1.0, full)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of frames")
    ap.add_argument("--live", action="store_true", help="use the real Lattice client (needs LATTICE_* env)")
    ap.add_argument("--json", dest="json_out", default=None, help="write mock wire records to this JSONL path")
    args = ap.parse_args(argv)

    publisher = LatticeEntityPublisher() if args.live else MockEntityPublisher()
    sc = run(args.records, args.aggregate, publisher, args.severity, args.limit)
    print(sc.render())

    if args.json_out and isinstance(publisher, MockEntityPublisher):
        with open(args.json_out, "w") as f:
            for rec in publisher.wire_records():
                f.write(json.dumps(rec, default=str) + "\n")
        print(f"\nwrote {len(publisher.published)} wire records to {args.json_out}")


if __name__ == "__main__":
    main()
