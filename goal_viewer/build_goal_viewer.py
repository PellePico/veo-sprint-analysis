"""
Build goal_viewer.html: a scrollable, frame-by-frame view of goals 1, 3 and 5,
from the moment home lose the ball until just after the goal.

Same conventions as starter.ipynb / demo_goal_moment.ipynb:
  - pitch 105 x 67.7 m, origin at the centre spot, same markings as draw_pitch()
  - x_att = x * attack_dir  ->  away always attack left -> right in the viewer
  - windows cut with >= / <= on timestamp_ms, like the demo notebook

"As exact as possible":
  - the ball comes from ball.csv (~30 Hz), the finest ball record
  - players are tracked every 400 ms; frames are drawn every 100 ms and player
    positions in between are linearly interpolated (marked "estimated")
  - jumps faster than 11 m/s between two tracking frames are flagged as
    suspect (tracking glitch / ID swap), drawn with a dashed outline

Run:  python goal_viewer/build_goal_viewer.py
      (reads data/ at the repo root, or the folder in $VEO_DATA)
"""
import base64
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("VEO_DATA", HERE.parent / "data"))
TEMPLATE = HERE / "template.html"
OUT = HERE / "goal_viewer.html"          # contains match data: gitignored, do not commit

events = pd.read_csv(DATA / "events.csv")
players = pd.read_parquet(DATA / "players.parquet")
ball = pd.read_csv(DATA / "ball.csv").sort_values("timestamp_ms")
players = players[players.player_id.notna()]
FRAMES = np.sort(players.timestamp_ms.unique())

STEP_MS = 100          # one viewer frame per 0.1 s
MAX_SPEED = 11.0       # m/s; faster than any footballer -> tracking glitch

# goal: (event_id of the goal, event_id where home lose the ball, why)
GOALS = {
    "1": (725, 720, "Home concede a free kick, away take it"),
    "3": (829, 824, "Ball goes out off home; away throw-in"),
    "5": (1012, 1006, "away_13 intercepts home_10's pass"),
}
LEAD_IN_MS = 2000      # start a little before the ball is lost
TAIL_MS = 1500         # keep going a little after the goal


# --- pitch control: simplified Spearman (2018) time-to-intercept model -------
# Each player keeps moving along his current velocity for a reaction time,
# then runs straight to the point at max speed. The team that can arrive
# sooner controls the point; the arrival-time gap is turned into a
# probability with a logistic curve (sigma = uncertainty in arrival time).
PC_NX, PC_NY = 53, 34          # grid: ~2 m cells
PC_REACT = 0.7                 # s
PC_VMAX = 5.0                  # m/s, running speed toward the point
PC_VMAX_CAP = 8.0              # m/s, cap on measured velocity (noise)
PC_SIGMA = 0.45                # s
_K = np.pi / (np.sqrt(3) * PC_SIGMA)
_gx = -52.5 + (np.arange(PC_NX) + 0.5) * 105.0 / PC_NX
_gy = 33.85 - (np.arange(PC_NY) + 0.5) * 67.7 / PC_NY      # row 0 = top (+y)
GRID = np.stack(np.meshgrid(_gx, _gy), -1).reshape(-1, 2)   # row-major, top row first


def _arrival(team):
    s = np.asarray(team, dtype=float)                       # (n, 4): x, y, vx, vy
    r = s[:, :2] + s[:, 2:] * PC_REACT                      # where reaction leaves them
    dist = np.linalg.norm(GRID[:, None, :] - r[None, :, :], axis=2)
    return (PC_REACT + dist / PC_VMAX).min(axis=1)


def pitch_control(home, away):
    """Base64 bytes, one per grid cell: 0 = away controls, 255 = home controls."""
    if not home or not away:
        return None
    p_home = 1 / (1 + np.exp(-_K * (_arrival(away) - _arrival(home))))
    return base64.b64encode(np.round(p_home * 255).astype(np.uint8).tobytes()).decode()


def ev(eid):
    return events[events.event_id == eid].iloc[0]


def describe(r):
    who = r.player_id if pd.notna(r.player_id) else f"{r.team} (player unknown)"
    name = r.event_name.replace("_", " ")
    if r.event_name == "pass" and pd.notna(r.receiver_player_id):
        return f"{name}: {who} → {r.receiver_player_id}"
    return f"{name}: {who}"


def build_goal(goal_eid, lost_eid):
    g, lost = ev(goal_eid), ev(lost_eid)
    d = int(g.attack_dir)                      # away's attack_dir this half
    t_start = int(lost.timestamp_ms - LEAD_IN_MS)
    t_end = int(g.timestamp_ms + TAIL_MS)

    # tracking window (one extra frame each side for interpolation)
    lo = FRAMES[FRAMES <= t_start].max()
    hi = FRAMES[FRAMES >= t_end].min()
    trk = players[(players.timestamp_ms >= lo) & (players.timestamp_ms <= hi)].copy()
    trk["xa"] = trk.x * d
    trk["ya"] = trk.y * d

    # flag impossible jumps between consecutive tracking frames
    trk = trk.sort_values(["player_id", "timestamp_ms"])
    step = np.hypot(trk.groupby("player_id").xa.diff(), trk.groupby("player_id").ya.diff())
    dt = trk.groupby("player_id").timestamp_ms.diff() / 1000
    trk["suspect"] = (step / dt) > MAX_SPEED
    # velocities for pitch control: central difference between neighbouring
    # tracking frames, glitches zeroed, capped at a sprint
    gp = trk.groupby("player_id")
    span = (gp.timestamp_ms.shift(-1).fillna(trk.timestamp_ms) - gp.timestamp_ms.shift(1).fillna(trk.timestamp_ms)) / 1000
    for col, v in (("xa", "vx"), ("ya", "vy")):
        delta = gp[col].shift(-1).fillna(trk[col]) - gp[col].shift(1).fillna(trk[col])
        trk[v] = (delta / span.replace(0, np.nan)).fillna(0.0)
    sp = np.hypot(trk.vx, trk.vy)
    scale = np.where(sp > PC_VMAX_CAP, PC_VMAX_CAP / sp.replace(0, 1), 1.0)
    trk["vx"] *= scale; trk["vy"] *= scale
    glitch = trk.suspect | gp.suspect.shift(-1).astype("boolean").fillna(False).astype(bool)
    trk.loc[glitch, ["vx", "vy"]] = 0.0
    by_t = {t: f.set_index("player_id") for t, f in trk.groupby("timestamp_ms")}
    win_frames = np.sort(np.array(list(by_t)))

    b = ball[(ball.timestamp_ms >= t_start - 200) & (ball.timestamp_ms <= t_end + 200)]
    bt, bx, by = b.timestamp_ms.values, (b.x * d).values, (b.y * d).values

    # timeline: every real tracking frame, plus 3 evenly spaced estimated
    # frames between each pair (~100 ms steps)
    real = win_frames[(win_frames >= t_start - 400) & (win_frames <= t_end + 400)]
    times = []
    for a_t, c_t in zip(real[:-1], real[1:]):
        n = max(1, round((c_t - a_t) / STEP_MS))
        times += [int(round(a_t + (c_t - a_t) * k / n)) for k in range(n)]
    times.append(int(real[-1]))

    frames = []
    for t in times:
        t0 = win_frames[win_frames <= t].max()
        t1 = win_frames[win_frames >= t].min()
        a, c = by_t[t0], by_t[t1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        exact = t1 == t0 or w in (0.0, 1.0)
        rows = []
        pc_players = {"home": [], "away": []}
        for pid in sorted(set(a.index) | set(c.index)):
            if pid in a.index and pid in c.index:
                x = a.at[pid, "xa"] * (1 - w) + c.at[pid, "xa"] * w
                y = a.at[pid, "ya"] * (1 - w) + c.at[pid, "ya"] * w
                vx = a.at[pid, "vx"] * (1 - w) + c.at[pid, "vx"] * w
                vy = a.at[pid, "vy"] * (1 - w) + c.at[pid, "vy"] * w
                flag = 2 if bool(c.at[pid, "suspect"]) else (0 if exact else 1)
                team = a.at[pid, "team"]; role = a.at[pid, "role"]
            else:                               # only in one frame: hold its position
                src = a if pid in a.index else c
                x, y = src.at[pid, "xa"], src.at[pid, "ya"]
                vx = vy = 0.0
                flag = 3
                team = src.at[pid, "team"]; role = src.at[pid, "role"]
            if team in pc_players:
                pc_players[team].append((x, y, vx, vy))
            tc = {"home": "h", "away": "a"}.get(team, "r")
            rows.append([pid.split("_")[-1], tc, round(float(x), 2), round(float(y), 2), flag,
                         1 if role == "goalkeeper" else 0])
        # ball: nearest ball.csv row within 70 ms, else unknown
        i = np.searchsorted(bt, t)
        cand = [j for j in (i - 1, i) if 0 <= j < len(bt)]
        j = min(cand, key=lambda k: abs(bt[k] - t)) if cand else None
        bpos = [round(float(bx[j]), 2), round(float(by[j]), 2)] if j is not None and abs(bt[j] - t) <= 70 else None
        frames.append({"t": t, "exact": bool(exact), "ball": bpos, "p": rows,
                       "pc": pitch_control(pc_players["home"], pc_players["away"])})

    evs = events[(events.timestamp_ms >= t_start) & (events.timestamp_ms <= t_end)]
    ev_list = [{"t": int(r.timestamp_ms), "id": int(r.event_id), "team": r.team, "text": describe(r)}
               for r in evs.itertuples()]
    return {
        "pc_grid": [PC_NX, PC_NY],
        "scorer": g.player_id, "t_goal": int(g.timestamp_ms), "t_lost": int(lost.timestamp_ms),
        "frames": frames, "events": ev_list,
    }


def build_lineups():
    """Kick-off frame + average positions for each half, away attacking left -> right."""
    out = {}
    for half in (1, 2):
        q = players[players.half == half].copy()
        d = int(q[q.team == "away"].attack_dir.iloc[0])
        q["xa"], q["ya"] = q.x * d, q.y * d
        ts = np.sort(q.timestamp_ms.unique())
        # kick-off: the most complete frame in the first 3 s of the half
        first = ts[ts <= ts[0] + 3000]
        counts = q[q.timestamp_ms.isin(first)].groupby("timestamp_ms").size()
        t_ko = int(counts.idxmax())
        ko = q[q.timestamp_ms == t_ko].set_index("player_id")
        rows = []
        for pid, r in ko.iterrows():
            rows.append([pid.split("_")[-1], {"home": "h", "away": "a"}.get(r.team, "r"),
                         round(float(r.xa), 2), round(float(r.ya), 2), 0, int(r.role == "goalkeeper")])
        # players missing at kick-off: borrow their nearest tracked position within 10 s
        n_side = ko[ko.team != "referee"].groupby("team").size().to_dict()
        for team in ("home", "away"):
            if n_side.get(team, 0) >= 11:
                continue
            near = q[(q.team == team) & (q.timestamp_ms <= t_ko + 60000) & ~q.player_id.isin(ko.index)]
            for pid, g in near.groupby("player_id"):
                if n_side.get(team, 0) >= 11:
                    break
                r = g.iloc[0]
                rows.append([pid.split("_")[-1], team[0], round(float(r.xa), 2), round(float(r.ya), 2), 3,
                             int(r.role == "goalkeeper")])
                n_side[team] = n_side.get(team, 0) + 1
        # average shape, split by who has the ball. Possession = team of the most
        # recent event, if it was within the last 5 s (otherwise unknown -> skipped)
        q = q[q.team != "referee"].sort_values("timestamp_ms")
        ev = events[(events.half == half) & events.team.notna()][["timestamp_ms", "team"]].sort_values("timestamp_ms")
        ev = ev.rename(columns={"team": "poss", "timestamp_ms": "t_ev"})
        q = pd.merge_asof(q, ev, left_on="timestamp_ms", right_on="t_ev", direction="backward")
        q = q[(q.timestamp_ms - q.t_ev) <= 5000]
        shapes = {}
        for name, mask in [("def", q.poss != q.team), ("att", q.poss == q.team)]:
            avg = q[mask].groupby(["player_id", "team", "role"]).agg(
                x=("xa", "mean"), y=("ya", "mean"), n=("timestamp_ms", "size")).reset_index()
            avg["mins"] = avg.n * 0.4 / 60
            avg = avg[avg.mins >= 3]
            shapes[name] = [[r.player_id.split("_")[-1], r.team[0], round(float(r.x), 2), round(float(r.y), 2), 0,
                             int(r.role == "goalkeeper"), round(float(r.mins), 1)] for r in avg.itertuples()]
        out[str(half)] = {"t": t_ko, "kickoff": rows, **shapes}
    # half-time changes
    subs = {}
    for team in ("home", "away"):
        a = set(players[(players.half == 1) & (players.team == team)].player_id)
        b = set(players[(players.half == 2) & (players.team == team)].player_id)
        num = lambda s: sorted((x.split("_")[-1] for x in s), key=int)
        subs[team] = {"off": num(a - b), "on": num(b - a)}
    out["subs"] = subs
    return out


payload = {}
for key, (goal_eid, lost_eid, why) in GOALS.items():
    payload[key] = build_goal(goal_eid, lost_eid)
    payload[key]["why"] = why
    print(f"goal {key}: {len(payload[key]['frames'])} frames, {len(payload[key]['events'])} events")

html = TEMPLATE.read_text()
lineups = build_lineups()
print("line-ups: kick-off frames", lineups["1"]["t"], lineups["2"]["t"])
html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
html = html.replace("/*__LINEUPS__*/null", json.dumps(lineups, separators=(",", ":")))
OUT.write_text(html)
print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")
