# Veo Football Analytics Sprint: why did the away side win?

A project from a **mini hackathon**: the **Veo Football Analytics Sprint** at Veo's offices in Copenhagen (9 November 2026). In one 90-minute sprint, mixed teams got one full 11-a-side match captured on a Veo camera (player tracking, ball tracking and detected on-ball events) and had to find something interesting in it. This repo is my part of that work, cleaned up afterwards.

## The question

On almost every process stat the two teams were level: shots, passes, pass completion, distance covered. But the away side won by two goals. **Why?**

My hypothesis: away didn't shoot *more*, they shot *better*. I tested it two ways:

| Step | Data | What it measures |
|---|---|---|
| 1. Shot location | events | median distance and angle to goal, share of shots inside the box |
| 2. Pressure on the shooter | tracking | distance to the nearest defender, and opponents in the cone from the shooter to the posts |
| 3. The goals, frame by frame | tracking + ball | how each goal developed, from the moment the home side lost the ball |

## What I found

- Away took their shots **much closer to goal** and **far more often from inside the box**, on a near-equal number of attempts.
- Their shooters also had **more space**: the nearest defender was further away and there were fewer bodies between ball and goal.
- So the scoreline is about **chance quality, not volume**. Both measures point the same way.

This describes one match of about 30 shots. It is not a fitted expected-goals (xG) model, which would need many matches.

## What's in the repo

```
notebooks/shot_quality.ipynb      steps 1 and 2: shot location and pressure
goal_viewer/build_goal_viewer.py  step 3: builds an interactive goal viewer (HTML)
goal_viewer/template.html         the viewer's page (data is injected at build time)
src/pitch.py                      shared helpers: loading, pitch drawing, frame lookup
```

### The goal viewer

`build_goal_viewer.py` turns three of the goals into a self-contained HTML page you can scrub through at 0.1-second steps:

- **Smoother movement:** players are tracked at 2.5 fps, so the frames in between are linearly interpolated and marked as estimated.
- **Exact ball:** the ball comes from the ~30 Hz ball track.
- **Glitch flags:** tracking glitches and ID swaps (jumps faster than 11 m/s) are flagged.
- **Defensive lines:** the home back line and the offside line are drawn, with their width and flatness.
- **Pitch control:** a simplified Spearman (2018) time-to-intercept model shows which team could reach each part of the pitch first.
- **Line-ups:** kick-off positions, plus average shapes in and out of possession.

## Data

**The dataset is not included.** Veo shared it with sprint participants on the condition that it isn't redistributed. The match is anonymised, and the built viewer (`goal_viewer.html`) is gitignored because it embeds the tracking data.

If you took part in the sprint, put your copy of the files in `data/`, or point `VEO_DATA` at the folder:

```
data/
  events.csv
  players.parquet
  ball.csv
```

## Run it

```bash
pip install -r requirements.txt

jupyter notebook notebooks/shot_quality.ipynb   # steps 1-2
python goal_viewer/build_goal_viewer.py         # step 3 -> goal_viewer/goal_viewer.html
```

## Possible next steps

- A proper xG model trained on many matches, so chance quality can be put in goals.
- Pressing and transitions: how quickly each side won the ball back, and where.
