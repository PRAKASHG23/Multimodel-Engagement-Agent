"""Activity 5.2 - Student session analytics (learner-facing).

Each summary describes ONE participant against their own session only. There
are no grades, rankings, leaderboards or peer comparisons.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def generate_student_analytics(attendance: list[dict], participant_scores: pd.DataFrame,
                               cfg: Config) -> list[dict]:
    result = []
    for rec in attendance:
        pid = rec["participant_id"]
        summary = {
            "participant_id": pid,
            "attendance": {k: rec[k] for k in ("status", "presence_seconds", "presence_percent",
                                              "first_seen", "last_seen")},
        }
        df = (participant_scores[participant_scores["participant_id"] == pid]
              .sort_values("window") if len(participant_scores) else pd.DataFrame())
        if df.empty:
            summary["engagement"] = None
            summary["notes"] = ["Not enough visible time to analyse engagement."]
            result.append(summary); continue

        scores = df["engagement"].tolist()
        mean = sum(scores) / len(scores)
        std = float(pd.Series(scores).std(ddof=0))
        consistency = max(0.0, 100 - 2 * std)  # 100 = perfectly steady
        focus = df[df["engagement"] >= cfg.focus_threshold]
        dips = df[df["engagement"] < cfg.dip_threshold]

        notes = []
        if len(scores) >= 4:
            half = len(scores) // 2
            first, second = sum(scores[:half]) / half, sum(scores[half:]) / (len(scores) - half)
            if second - first >= 10:
                notes.append("Your focus built up over the session.")
            elif first - second >= 10:
                notes.append("Your focus faded towards the end of the session.")
        if consistency >= 80:
            notes.append("Your engagement was steady throughout.")
        if len(focus):
            notes.append(f"Strongest focus around {_fmt(focus['t_start'].iloc[0])}.")
        if len(dips):
            notes.append(f"A dip in activity around {_fmt(dips['t_start'].iloc[0])}; "
                         "this can be normal after long stretches of concentration.")
        if not notes:
            notes.append("Engagement was moderate with no notable swings.")

        summary["engagement"] = {
            "average": round(mean, 1),
            "consistency": round(consistency, 1),
            "focused_windows": int(len(focus)),
            "dip_windows": int(len(dips)),
            "window_seconds": cfg.window_seconds,
            "timeline": [{"t_start": float(r.t_start), "engagement": float(r.engagement)}
                         for r in df.itertuples()],
        }
        summary["notes"] = notes
        result.append(summary)
    return result
