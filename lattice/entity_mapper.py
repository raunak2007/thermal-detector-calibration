"""Map a thermal-detector detection onto a real Lattice Entity, with a trust gate.

The point of this module is not "call the SDK". It is the thing the calibration
study forces you to confront once you try to publish detections into a shared
picture:

  1. Lattice has no confidence field. The only `confidence` in the whole schema
     is on emitter_notation (SIGINT). So you cannot just hang the detector's
     score on the entity, and you would not want to: the study showed the score
     is miscalibrated under degradation and in the under-confident direction, so
     the number means less exactly when the scene is worst.

  2. The detector fails by going blind, not by lying. Under the hard
     degradations recall collapses from 96% to under 6%. An empty map from a
     blind sensor looks identical to an empty map from a clear field. The
     integration's job is to make those two distinguishable.

So the mapping does three things a generic SDK demo would not:

  - a trust gate decides, per frame, whether the detector is inside the
    reliability envelope the benchmark measured (driven by the measured mAP at
    that condition, not by the per-box score);
  - a trusted detection becomes an Entity whose `mil_view.disposition` asserts
    only as hard as the regime earns (SUSPICIOUS when trusted, PENDING when
    degraded), with the raw score carried as human-readable text, never as a
    fake confidence field;
  - when the gate says the producer is blind, individual boxes are suppressed
    and the producer's own `health` flips to FAIL instead, so the operator sees
    "this sensor went dark" rather than a quiet, sparse, untrustworthy scatter.

Two honest constraints are wired in rather than hidden:

  - No geolocation. records.csv does not retain pixel boxes, and HIT-UAV ships
    no camera pose or altitude, so there is no way to turn a detection into a
    real lat/lon. `demo_location` returns a labelled placeholder anchor with a
    deliberately huge error ellipse that says, in the schema's own terms, that
    the position is not to be trusted. A real deployment fills this from camera
    intrinsics + platform pose + a ground model.
  - Live Lattice is credential-gated. This module only builds Entity objects
    from the real SDK types; publish.py mocks the transport by default.
"""
from __future__ import annotations

import datetime as dt
import math
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from anduril import (
    Aliases,
    ErrorEllipse,
    Health,
    Indicators,
    Location,
    LocationUncertainty,
    MilView,
    Ontology,
    Position,
    Provenance,
)

INTEGRATION_NAME = "thermal-detector-calibration"

# Reliability envelope, read as the measured mAP@0.5 (0-100) at the frame's
# condition from the benchmark's aggregate.csv. In the field this lookup would
# be fed by a runtime scene/degradation classifier; here the condition label is
# known, so the gate is backed directly by the offline reliability measurement.
# The thresholds are where the measured sweep actually breaks: clean sits at 92,
# the dead control at 88, and the three hard degradations fall into single
# digits at full severity.
GATE_TRUSTED_MIN_MAP = 80.0
GATE_DEGRADED_MIN_MAP = 30.0

# Labelled placeholder. NOT a geolocation. See module docstring.
DEMO_ANCHOR_LAT = 34.17
DEMO_ANCHOR_LON = -117.90
DEMO_ANCHOR_ALT_M = 120.0
_ENTITY_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "github.com/raunak2007/thermal-detector-calibration")


class Regime(str, Enum):
    TRUSTED = "TRUSTED"
    DEGRADED = "DEGRADED"
    BLIND = "BLIND"


@dataclass
class Detection:
    """One row of the benchmark's per-detection log (results_thermal/records.csv)."""
    cls: int
    score: float
    image_id: str
    perturbation: str
    severity: float
    is_tp: Optional[int] = None


def regime_for_reliability(map_at_condition: float) -> Regime:
    """Classify the detector's operating regime from its measured mAP at the condition."""
    if map_at_condition >= GATE_TRUSTED_MIN_MAP:
        return Regime.TRUSTED
    if map_at_condition >= GATE_DEGRADED_MIN_MAP:
        return Regime.DEGRADED
    return Regime.BLIND


def disposition_for_regime(regime: Regime) -> str:
    """How hard to assert the track. A detected person is unknown intent, never
    FRIENDLY/HOSTILE from a thermal blob; the regime only lowers the assertion.

      TRUSTED  -> SUSPICIOUS  (a person of unknown intent, asserted)
      DEGRADED -> PENDING     (producer partly reliable; assert tentatively)
      BLIND    -> UNKNOWN     (only reached if a caller forces publication)
    """
    return {
        Regime.TRUSTED: "DISPOSITION_SUSPICIOUS",
        Regime.DEGRADED: "DISPOSITION_PENDING",
        Regime.BLIND: "DISPOSITION_UNKNOWN",
    }[regime]


def health_status_for_regime(regime: Regime) -> str:
    """The producer's own health. This is the channel that carries the blind
    failure: a quiet detector reports FAIL here instead of going silent."""
    return {
        Regime.TRUSTED: "HEALTH_STATUS_HEALTHY",
        Regime.DEGRADED: "HEALTH_STATUS_WARN",
        Regime.BLIND: "HEALTH_STATUS_FAIL",
    }[regime]


def _stable_entity_id(*parts: object) -> str:
    return str(uuid.uuid5(_ENTITY_NAMESPACE, "|".join(str(p) for p in parts)))


def demo_location(image_id: str, det_index: int) -> tuple[Location, LocationUncertainty]:
    """Placeholder position, NOT geolocation. Spreads detections a few hundred
    metres around a demo anchor so the Lattice map is not a single stacked pin,
    and attaches a 1 km error ellipse so the uncertainty states, in the schema's
    own terms, that the position is synthetic. A real integration replaces this
    with camera intrinsics + platform pose + ground intersection."""
    h = uuid.uuid5(_ENTITY_NAMESPACE, f"{image_id}#{det_index}").int
    bearing = (h % 360) * math.pi / 180.0
    radius_m = 80.0 + (h // 360) % 300
    dlat = (radius_m * math.cos(bearing)) / 111_320.0
    dlon = (radius_m * math.sin(bearing)) / (111_320.0 * math.cos(math.radians(DEMO_ANCHOR_LAT)))
    loc = Location(position=Position(
        latitude_degrees=DEMO_ANCHOR_LAT + dlat,
        longitude_degrees=DEMO_ANCHOR_LON + dlon,
        altitude_hae_meters=DEMO_ANCHOR_ALT_M,
    ))
    unc = LocationUncertainty(position_error_ellipse=ErrorEllipse(
        probability=0.1, semi_major_axis_m=1000.0, semi_minor_axis_m=1000.0, orientation_d=0.0,
    ))
    return loc, unc


def _provenance(image_id: str, now: dt.datetime) -> Provenance:
    return Provenance(
        integration_name=INTEGRATION_NAME,
        data_type="EO/IR thermal person detection (YOLOv8n, HIT-UAV)",
        source_id=image_id,
        source_update_time=now,
    )


def detection_to_entity_kwargs(
    det: Detection, regime: Regime, det_index: int, now: Optional[dt.datetime] = None,
) -> dict:
    """Build the publish_entity(**kwargs) for one detected person.

    The raw score rides in the display name and description as text, flagged with
    the regime, because there is no honest confidence field to put it in and the
    study says the number is not trustworthy under degradation anyway.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    loc, unc = demo_location(det.image_id, det_index)
    score_note = f"score {det.score:.2f} (raw detector confidence, uncalibrated)"
    return dict(
        entity_id=_stable_entity_id("track", det.image_id, det_index),
        is_live=True,
        expiry_time=now + dt.timedelta(minutes=5),
        description=(
            f"Thermal person detection, {regime.value} regime. {score_note}. "
            f"DEMO position (not geolocated)."
        ),
        location=loc,
        location_uncertainty=unc,
        aliases=Aliases(name=f"person [{regime.value.lower()}] {det.score:.2f}"),
        mil_view=MilView(
            disposition=disposition_for_regime(regime),
            environment="ENVIRONMENT_LAND",
        ),
        ontology=Ontology(platform_type="Person", specific_type="person-thermal-ir"),
        provenance=_provenance(det.image_id, now),
        indicators=Indicators(simulated=True),
    )


def sensor_health_entity_kwargs(
    image_id: str, perturbation: str, severity: float, regime: Regime,
    map_at_condition: float, n_suppressed: int = 0, now: Optional[dt.datetime] = None,
) -> dict:
    """Build the publish_entity(**kwargs) for the producer's own status.

    This is the degradation-as-signal entity. It is one persistent entity (the
    sensor platform), updated per frame, whose health reflects the regime. On a
    blind frame it reports FAIL and says how many detections it chose to
    suppress, so the operator never mistakes a blind sensor for a clear field.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    status = health_status_for_regime(regime)
    if regime is Regime.BLIND:
        human = (f"THERMAL DETECTOR BLIND under {perturbation} (sev {severity:g}): "
                 f"measured mAP {map_at_condition:.0f}/100, recall collapsed. "
                 f"{n_suppressed} low-trust detection(s) suppressed this frame.")
    elif regime is Regime.DEGRADED:
        human = (f"Thermal detector degraded under {perturbation} (sev {severity:g}): "
                 f"measured mAP {map_at_condition:.0f}/100. Tracks asserted tentatively.")
    else:
        human = f"Thermal detector nominal (measured mAP {map_at_condition:.0f}/100)."
    return dict(
        entity_id=_stable_entity_id("producer", INTEGRATION_NAME),
        is_live=True,
        expiry_time=now + dt.timedelta(minutes=5),
        description=human,
        aliases=Aliases(name="Thermal detector (YOLOv8n/HIT-UAV)"),
        ontology=Ontology(platform_type="Sensor", specific_type="eo-ir-thermal-detector"),
        provenance=_provenance(image_id, now),
        health=Health(health_status=status, update_time=now),
        indicators=Indicators(simulated=True, starred=(regime is Regime.BLIND)),
    )
