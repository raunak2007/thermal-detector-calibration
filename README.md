# welfare-harness

Pilot for SPAR Project 3. Holds the model fixed, varies the harness, and places
each condition on a **disagreement / utility** plot. No GPUs, no finetuning:
every intervention sits above the model.

    x = disagreement   spread of welfare judgments ACROSS raters
    y = utility        decision accuracy

The result worth having is an intervention that sits **up and to the left** of
baseline: less disagreement, no utility cost.

## Run

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...
./find_materials.sh                  # look for Caviola's validated instrument
python run_pilot.py --out runs/p1.json --tasks 10
python analyze.py runs/p1.json       # prints the table, writes pareto.png
```

## Layout

| file | what it is |
|---|---|
| `find_materials.sh` | OSF API sweep for Caviola supplementary materials |
| `items/welfare_items.json` | disagreement axis, 10 items — **placeholder wording** |
| `items/decision_tasks.json` | utility axis, 10 checkable decision tasks |
| `conditions.py` | the four harness conditions and the rater panel |
| `run_pilot.py` | generates transcripts, collects ratings, grades tasks |
| `analyze.py` | disagreement vs utility table and plot |

## Conditions

| name | kind | what changes |
|---|---|---|
| `baseline` | baseline | stateless, neutral system prompt |
| `memory` | functional | prior session in context, memory claimed |
| `persona` | presentational | named persona, no state added |
| `disclosure` | intervention | accuracy about its own nature |

One functional and one presentational change so the distinction is visible in
the first table. `disclosure` is the candidate intervention, in the spirit of
the safety-nudges line of work.

## Things that are placeholders, not findings

- **Item wording is mine, not Caviola's.** The constructs come from what was
  named in the meeting. Swap in validated wording as soon as `find_materials.sh`
  or an email to Caviola turns it up, and note the change in the writeup.
- **Raters are models, not people.** That is deliberate for the pilot: it
  sidesteps IRB while the instrument is still moving. Disagreement among five
  model families is not disagreement among humans, and the paper will need the
  human version. Start the IRB conversation before that becomes the blocker.
- **Ten tasks is a pilot floor.** Accuracy on ten items has a confidence
  interval wide enough to hide most real effects. Scale before claiming a
  utility cost or the absence of one.
- `t09` in the task set has a worked rationale that contradicts its own
  answer. Fix it before running, or drop the item.

## A secondary measure worth adding

Resample each self-report probe 20–30 times per condition and record the
spread of the model's *own* answers, not just the raters'. Shanahan et al.
use exactly this to separate a stable property from one generated on the fly.
If unstable self-reports drive rater disagreement, that is a mechanism linking
the harness to the outcome rather than just a correlation.
