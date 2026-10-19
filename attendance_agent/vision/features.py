"""Per-window visual indicators: presence, head-pose proxy, movement, inactivity."""
from __future__ import annotations

import math
from collections import defaultdict

from ..config import Config
from .tracker import Observation


def extract_window_features(observations: list[Observation], sample_dt: float,
                            duration: float, cfg: Config) -> list[dict]:
    win = cfg.window_seconds
    n_windows = max(1, math.ceil(duration / win))
    buckets: dict[tuple[int, int], list[Observation]] = defaultdict(list)
    for o in observations:
        w = min(int(o.t // win), n_windows - 1)
        buckets[(w, o.track_id)].append(o)

    rows = []
    for (w, pid), obs in sorted(buckets.items()):
        obs.sort(key=lambda o: o.t)
        t0, t1 = w * win, min((w + 1) * win, duration)
        span = max(t1 - t0, sample_dt)
        presence = min(1.0, len(obs) * sample_dt / span)
        facing = sum(o.pose == "frontal" for o in obs) / len(obs)

        speeds = []
        for a, b in zip(obs, obs[1:]):
            dt = b.t - a.t
            if 0 < dt <= cfg.occlusion_tolerance_s:
                size = max((a.w + b.w) / 2, 1.0)
                speeds.append(math.hypot(b.cx - a.cx, b.cy - a.cy) / size / dt)
        movement = sum(speeds) / len(speeds) if speeds else 0.0
        inactivity = sum(s < cfg.still_speed for s in speeds) / len(speeds) if speeds else 1.0
        rows.append({
            "window": w, "t_start": t0, "t_end": t1,
            "participant_id": f"P{pid}",
            "presence_fraction": round(presence, 3),
            "facing_ratio": round(facing, 3),
            "movement": round(movement, 3),
            "inactivity_ratio": round(inactivity, 3),
        })
    return rows
