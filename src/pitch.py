"""Shared helpers: data loading, pitch drawing and frame lookup.

Coordinates follow the Veo sprint dataset: metres, origin at the centre spot,
x along the pitch, y across it. Teams swap ends at half-time, so every
"attacking" coordinate is the raw one multiplied by `attack_dir`.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PITCH_L, PITCH_W = 105.0, 67.7
COLOURS = {"home": "#1f77b4", "away": "#d62728", "referee": "#999999"}

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VEO_DATA", REPO / "data"))


def load(data_dir=DATA):
    """Load events + tracking and add attack-direction coordinates."""
    data_dir = Path(data_dir)
    missing = [f for f in ("events.csv", "players.parquet") if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not found in {data_dir}. The dataset is not part of this repo; "
            "put your own copy in data/ or point VEO_DATA at it."
        )
    events = pd.read_csv(data_dir / "events.csv")
    players = pd.read_parquet(data_dir / "players.parquet")
    for df, (x, y) in ((events, ("x_start", "y_start")), (players, ("x", "y"))):
        df["x_att"] = df[x] * df.attack_dir
        df["y_att"] = df[y] * df.attack_dir
    return events, players


class Frames:
    """Nearest-frame lookup on the tracking data (tracking runs at 2.5 fps)."""

    def __init__(self, players):
        self.players = players
        self.times = np.sort(players.timestamp_ms.unique())

    def at(self, ms):
        t = self.times[np.abs(self.times - ms).argmin()]
        return self.players[self.players.timestamp_ms == t]


def video_time(ms):
    """timestamp_ms -> 'mm:ss' on the match video (same clock as the data)."""
    s = int(ms // 1000)
    return f"{s // 60}:{s % 60:02d}"


def draw_pitch(ax=None, figsize=(12, 7.5)):
    """Empty pitch with the standard markings."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    hl, hw = PITCH_L / 2, PITCH_W / 2
    ax.add_patch(plt.Rectangle((-hl, -hw), PITCH_L, PITCH_W, fill=False, color="#555"))
    ax.plot([0, 0], [-hw, hw], color="#555")
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="#555"))
    for s in (-1, 1):
        ax.add_patch(plt.Rectangle((s * hl, -20.15), -s * 16.5, 40.3, fill=False, color="#555"))
        ax.add_patch(plt.Rectangle((s * hl, -9.16), -s * 5.5, 18.32, fill=False, color="#555"))
    ax.set_xlim(-hl - 3, hl + 3)
    ax.set_ylim(-hw - 3, hw + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_frame(frame, ax=None, title=None):
    """Draw all players and the ball for one tracking frame."""
    ax = draw_pitch(ax)
    for team, g in frame.groupby("team"):
        ax.scatter(g.x, g.y, s=220, c=COLOURS[team], edgecolor="white", zorder=3, label=team)
        for r in g.dropna(subset=["player_id"]).itertuples():
            ax.text(r.x, r.y, r.player_id.split("_")[-1], ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold", zorder=4)
    b = frame.iloc[0]
    if pd.notna(b.ball_x):
        ax.scatter(b.ball_x, b.ball_y, s=90, c="white", edgecolor="black", lw=2, zorder=5)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower center", ncol=3, frameon=False)
    return ax
