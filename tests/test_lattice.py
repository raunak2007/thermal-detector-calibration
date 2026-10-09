"""The Lattice mapping is driven by the measured reliability, lands on real
schema fields, and suppresses-plus-flags rather than going silent when blind."""

import datetime as dt

import pandas as pd
from anduril import Entity

from lattice.entity_mapper import (
    Detection, Regime, detection_to_entity_kwargs, disposition_for_regime,
    health_status_for_regime, regime_for_reliability, sensor_health_entity_kwargs,
)
from lattice.publish import MockEntityPublisher, run

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def test_gate_thresholds_track_the_measured_sweep():
    assert regime_for_reliability(92.2) is Regime.TRUSTED   # clean
    assert regime_for_reliability(88.1) is Regime.TRUSTED    # dead control
    assert regime_for_reliability(36.0) is Regime.DEGRADED   # low_contrast @ full
    assert regime_for_reliability(4.4) is Regime.BLIND       # sensor_noise @ full
    assert regime_for_reliability(0.26) is Regime.BLIND      # atmospheric_blur @ full


def test_disposition_and_health_only_soften_with_the_regime():
    assert disposition_for_regime(Regime.TRUSTED) == "DISPOSITION_SUSPICIOUS"
    assert disposition_for_regime(Regime.DEGRADED) == "DISPOSITION_PENDING"
    assert health_status_for_regime(Regime.TRUSTED) == "HEALTH_STATUS_HEALTHY"
    assert health_status_for_regime(Regime.BLIND) == "HEALTH_STATUS_FAIL"


def test_track_maps_to_a_valid_entity_with_no_fake_confidence_field():
    d = Detection(0, 0.83, "f.jpg", "clean", 0.0, is_tp=1)
    kw = detection_to_entity_kwargs(d, Regime.TRUSTED, 0, now=NOW)
    e = Entity(**kw)  # real pydantic validation
    wire = e.model_dump(by_alias=True, exclude_none=True)
    assert wire["milView"]["disposition"] == "DISPOSITION_SUSPICIOUS"
    assert wire["ontology"]["platformType"] == "Person"
    assert wire["indicators"]["simulated"] is True
    # the score is carried as text, never as a numeric confidence field
    assert "0.83" in wire["aliases"]["name"]
    assert "confidence" not in wire and "score" not in wire


def test_blind_producer_health_is_fail_and_names_the_suppression():
    kw = sensor_health_entity_kwargs("f.jpg", "sensor_noise", 1.0, Regime.BLIND, 4.4,
                                     n_suppressed=7, now=NOW)
    e = Entity(**kw)
    wire = e.model_dump(by_alias=True, exclude_none=True)
    assert wire["health"]["healthStatus"] == "HEALTH_STATUS_FAIL"
    assert "BLIND" in wire["description"] and "7" in wire["description"]


def test_run_suppresses_tracks_on_blind_frames_and_emits_one_fail_signal(tmp_path):
    # two frames: one clean (trusted), one at a blind condition, each with a real + ghost det
    records = pd.DataFrame([
        dict(cls=0, score=0.9, is_tp=1, image_id="a", perturbation="clean", severity=1.0),
        dict(cls=0, score=0.3, is_tp=0, image_id="a", perturbation="clean", severity=1.0),
        dict(cls=0, score=0.9, is_tp=1, image_id="b", perturbation="sensor_noise", severity=1.0),
        dict(cls=0, score=0.3, is_tp=0, image_id="b", perturbation="sensor_noise", severity=1.0),
    ])
    agg = pd.DataFrame([
        dict(perturbation="clean", severity=1.0, map=92.2),
        dict(perturbation="sensor_noise", severity=1.0, map=4.4),
    ])
    rp, ap = tmp_path / "records.csv", tmp_path / "aggregate.csv"
    records.to_csv(rp, index=False); agg.to_csv(ap, index=False)

    pub = MockEntityPublisher()
    sc = run(str(rp), str(ap), pub, severity=1.0, now=NOW)

    assert sc.frames_by_regime["TRUSTED"] == 1
    assert sc.frames_by_regime["BLIND"] == 1
    assert sc.tracks_published == 2          # only the clean frame's two detections
    assert sc.suppressed_total == 2          # the blind frame's two detections
    assert sc.suppressed_real == 1           # one of them was a real person
    assert sc.health_signals["BLIND"] == 1   # exactly one FAIL signal in their place
    # every published object is a real, valid Entity
    assert all(isinstance(e, Entity) for e in pub.published)
