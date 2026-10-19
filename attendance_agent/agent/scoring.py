"""Activity 4.3 - Engagement scoring logic.

Transparent, rule-based scoring on a 0-100 scale. Every score carries its
component values and plain-language reasons, so the result is explainable.

Components (all 0..1):
  presence   - how much of the window the participant was visible
  attention  - share of time facing the camera (head-pose proxy)
  activity   - penalises long inactivity and excessive fidgeting
  audio      - healthy speech interaction in a not-too-noisy room (class level)
"""
from __future__ import annotations

from ..config import Config


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class EngagementScorer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def audio_component(self, noise_db: float, speech_ratio: float) -> float:
        c = self.cfg
        interaction = _clamp(speech_ratio / c.target_speech_ratio)
        # 0 penalty at/below quiet_db, full penalty at/above noisy_db
        noise_penalty = _clamp((noise_db - c.quiet_db) / (c.noisy_db - c.quiet_db))
        return _clamp(0.7 * interaction + 0.3 * (1 - noise_penalty))

    def activity_component(self, movement: float, inactivity: float) -> float:
        c = self.cfg
        score = 1.0 - 0.8 * inactivity
        if movement > c.fidget_speed:
            score *= max(0.3, c.fidget_speed / movement)
        return _clamp(score)

    def score(self, vis: dict, aud: dict | None):
        """Return (score, components, reasons) or None if not enough presence."""
        c = self.cfg
        if vis["presence_fraction"] < c.min_window_presence:
            return None
        comp = {
            "presence": _clamp(vis["presence_fraction"]),
            "attention": _clamp(vis["facing_ratio"]),
            "activity": self.activity_component(vis["movement"], vis["inactivity_ratio"]),
        }
        has_audio = (aud is not None and aud.get("noise_db") == aud.get("noise_db")
                     and aud.get("speech_ratio") == aud.get("speech_ratio"))  # NaN check
        if has_audio:
            comp["audio"] = self.audio_component(aud["noise_db"], aud["speech_ratio"])
            weights = c.weights_with_audio
        else:
            weights = c.weights_visual_only
        total = sum(weights.values())
        score = 100 * sum(weights[k] * comp[k] for k in weights) / total

        reasons = []
        if comp["attention"] < 0.5:
            reasons.append("often looking away from the front")
        elif comp["attention"] >= 0.85:
            reasons.append("consistently facing forward")
        if vis["inactivity_ratio"] > 0.9:
            reasons.append("very little movement")
        if vis["movement"] > c.fidget_speed:
            reasons.append("restless movement")
        if has_audio:
            if aud["noise_db"] >= c.noisy_db:
                reasons.append("noisy room")
            if aud["speech_ratio"] < 0.05:
                reasons.append("little spoken interaction")
            elif aud["speech_ratio"] >= c.target_speech_ratio:
                reasons.append("healthy spoken interaction")
        return round(score, 1), {k: round(v, 3) for k, v in comp.items()}, reasons

    def label(self, score: float) -> str:
        if score < self.cfg.low_engagement:
            return "low"
        if score >= self.cfg.high_engagement:
            return "high"
        return "moderate"
