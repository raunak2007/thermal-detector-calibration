# Publishing thermal detections into Lattice, through a trust gate

This takes the detector from the benchmark one level up the stack: into a shared
operational picture. It maps detections onto real [Anduril Lattice](https://developer.anduril.com)
`Entity` objects using the official `anduril-lattice-sdk`, but the interesting
part is not the SDK call. It is what the calibration result forces you to do
before you are allowed to make it.

## The problem the benchmark hands you

The main study (see the repo's one-pager) found two things that matter the
moment you try to put detections on a map:

1. **There is no honest number to publish.** Lattice has no detection-confidence
   field anywhere in its Entity schema. The only `confidence` in the whole SDK
   is on a SIGINT type. And the study showed you would not want to publish the
   detector's score even if there were a field for it, because under degradation
   the score is badly miscalibrated and in the under-confident direction, so it
   means least exactly when the scene is worst.

2. **The detector fails by going blind, not by lying.** Under the hard
   degradations recall falls from 96% to under 6%. The danger is silence: an
   empty map from a blind sensor looks identical to an empty map from a clear
   field.

So a generic "detections to entities" shim would be actively unsafe here. It
would stream a miscalibrated number into a field that does not exist, and on the
worst frames it would quietly publish almost nothing, which reads to an operator
as "area clear".

## What this does instead

`entity_mapper.py` carries the calibration result into the mapping on fields
that actually exist in the schema:

- **A trust gate** classifies each frame's regime from the detector's *measured*
  mAP at that condition (read from the benchmark's `aggregate.csv`), not from the
  per-box score. Trusted at mAP >= 80, degraded down to 30, blind below that. In
  a real deployment the gate is fed by a runtime degradation detector; here the
  condition is known, so the gate is backed directly by the offline reliability
  measurement, which is the honest version of the same thing.

- **Disposition only asserts as hard as the regime earns.** A trusted person
  detection publishes as `DISPOSITION_SUSPICIOUS` (a person of unknown intent,
  never friendly or hostile from a thermal blob). A degraded one drops to
  `DISPOSITION_PENDING`. The raw score rides along as text in the entity name and
  description, explicitly labelled uncalibrated, because there is no confidence
  field and the study says not to trust the number anyway.

- **Blind frames flip the producer's health instead of going silent.** When the
  gate says the detector is blind, the individual boxes are suppressed and the
  producer's own `Entity` reports `HEALTH_STATUS_FAIL` with a description of how
  many detections it chose to drop. The operator sees "this sensor went dark",
  which is the one thing a silent detector cannot tell them.

That last move is the whole point. The failure mode the benchmark found is
silence, so the right product behaviour is to make the silence loud.

## Run it

```bash
pip install anduril-lattice-sdk        # the real SDK; also in requirements.txt
python -m lattice.publish --severity 1.0     # full-severity frames, the regime split
python -m lattice.publish --severity 0.0     # clean frames, all trusted
python -m lattice.publish --json wire.jsonl  # dump the mock wire records to inspect
```

The default transport is a mock that validates every entity against the real SDK
types and records what would go on the wire, so it runs with no credentials. It
prints a scorecard, not a success line: how many frames fell into each regime,
how many tracks were asserted and how many of those were real people vs ghosts,
and how many real people were suppressed on blind frames. That suppression is the
honest cost of the gate, so it is reported rather than hidden.

`--live` uses the real `client.entities.publish_entity` call instead, and needs
`LATTICE_BASE_URL` and `LATTICE_TOKEN` in the environment.

## What is real and what is stubbed

Honest about the seams, since that is the point of the whole repo:

- **Real:** the SDK types and the field mapping. Every entity is constructed and
  validated by the real `anduril` pydantic models and serializes to the correct
  Lattice wire form (`tests/test_lattice.py` checks this). The `--live` path uses
  the real client method.
- **Stubbed, and labelled:** geolocation and the transport. `records.csv` does
  not keep pixel boxes, and HIT-UAV ships no camera pose or altitude, so there is
  no way to turn a detection into a true lat/lon. `demo_location` returns a
  placeholder anchor with a deliberately huge (1 km) error ellipse, so the
  position's own uncertainty field says, in the schema's terms, that it is not
  real. The runtime degradation detector that would feed the gate in the field is
  stood in for by the known condition label. Live Lattice is credential-gated, so
  the transport is mocked by default.
