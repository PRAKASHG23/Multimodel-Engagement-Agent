"""Activity 2.2 - Presence-based attendance.

Attendance is derived from cumulative presence time, never a single frame.
Intermittent appearances accumulate; a participant is marked present only once
the total passes ``min_presence_seconds``.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from ..config import Config
from .tracker import Observation


def compute_attendance(observations: list[Observation], sample_dt: float,
                       session_duration: float, cfg: Config,
                       session_start: datetime | None = None) -> list[dict]:
    by_id: dict[int, list[Observation]] = defaultdict(list)
    for o in observations:
        by_id[o.track_id].append(o)

    def stamp(t: float) -> str:
        if session_start is None:
            return f"{t:.1f}s"
        return (session_start + timedelta(seconds=t)).isoformat(timespec="seconds")

    records = []
    for pid in sorted(by_id):
        obs = sorted(by_id[pid], key=lambda o: o.t)
        presence = sample_dt  # the first observation itself
        for prev, cur in zip(obs, obs[1:]):
            gap = cur.t - prev.t
            # brief occlusions are bridged; long absences only credit the sample itself
            presence += gap if gap <= cfg.occlusion_tolerance_s else sample_dt
        records.append({
            "participant_id": f"P{pid}",
            "first_seen_s": round(obs[0].t, 2),
            "last_seen_s": round(obs[-1].t, 2),
            "first_seen": stamp(obs[0].t),
            "last_seen": stamp(obs[-1].t),
            "presence_seconds": round(presence, 1),
            "presence_percent": round(100 * min(presence, session_duration) / session_duration, 1)
            if session_duration > 0 else 0.0,
            "status": "present" if presence >= cfg.min_presence_seconds else "absent",
        })
    return records
