"""Activity 4.1 / 4.2 - Multimodal agent and signal fusion.

Aligns vision and audio by time window and produces
  * participant_scores - one row per participant per window
  * class_timeline     - one row per window (class-level engagement + audio context)
Either modality may be missing for some windows; the agent degrades gracefully
(visual-only scoring) rather than failing.
"""
from __future__ import annotations

import math

import pandas as pd

from ..config import Config
from .scoring import EngagementScorer


class MultimodalAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scorer = EngagementScorer(cfg)

    def fuse(self, visual_rows: list[dict], audio_df: pd.DataFrame | None,
             duration: float):
        cfg = self.cfg
        n_win = max(1, math.ceil(duration / cfg.window_seconds))
        audio_by_win = {}
        if audio_df is not None:
            audio_by_win = {int(r["window"]): r for r in audio_df.to_dict("records")}

        scored = []
        for v in visual_rows:
            aud = audio_by_win.get(v["window"])
            res = self.scorer.score(v, aud)
            if res is None:
                continue
            score, comp, reasons = res
            scored.append({
                **v,
                "noise_db": aud["noise_db"] if aud else float("nan"),
                "speech_ratio": aud["speech_ratio"] if aud else float("nan"),
                "engagement": score,
                "level": self.scorer.label(score),
                "components": comp,
                "reasons": "; ".join(reasons),
            })
        participant_scores = pd.DataFrame(scored)

        timeline = []
        for w in range(n_win):
            part = participant_scores[participant_scores["window"] == w] if scored else pd.DataFrame()
            a = audio_by_win.get(w)
            timeline.append({
                "window": w,
                "t_start": w * cfg.window_seconds,
                "t_end": min((w + 1) * cfg.window_seconds, duration),
                "participants": int(len(part)),
                "class_engagement": round(float(part["engagement"].mean()), 1) if len(part) else float("nan"),
                "noise_db": a["noise_db"] if a else float("nan"),
                "speech_ratio": a["speech_ratio"] if a else float("nan"),
                "audio_activity": a["audio_activity"] if a else "no_audio",
            })
        class_timeline = pd.DataFrame(timeline)
        class_timeline["level"] = class_timeline["class_engagement"].map(
            lambda s: self.scorer.label(s) if s == s else "no_data")
        return participant_scores, class_timeline
