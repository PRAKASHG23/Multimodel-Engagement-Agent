"""Activity 5.1 - Teaching suggestion logic (instructor-facing).

Rule-based, grounded in observable trends. Every suggestion names the time
range and the evidence behind it. These are assistive hints, never decisions.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config


def _runs(mask: list[bool]):
    """Yield (start, end_inclusive) for each run of True."""
    start = None
    for i, m in enumerate(mask + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            yield start, i - 1
            start = None


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def generate_suggestions(timeline: pd.DataFrame, cfg: Config) -> list[dict]:
    out: list[dict] = []
    if timeline is None or timeline.empty:
        return out
    tl = timeline.reset_index(drop=True)
    eng = tl["class_engagement"].tolist()
    win = cfg.window_seconds

    def span(a, b):
        return f"{_fmt(tl.loc[a, 't_start'])}-{_fmt(tl.loc[b, 't_end'])}"

    def add(kind, priority, a, b, title, detail, evidence):
        out.append({"type": kind, "priority": priority, "from_s": float(tl.loc[a, "t_start"]),
                    "to_s": float(tl.loc[b, "t_end"]), "time_range": span(a, b),
                    "suggestion": title, "why": detail, "evidence": evidence})

    # 1. Sustained low engagement -> interactive activity
    low = [(e == e and e < cfg.low_engagement) for e in eng]
    for a, b in _runs(low):
        if b - a + 1 >= cfg.sustained_low_windows:
            m = sum(eng[a:b + 1]) / (b - a + 1)
            add("sustained_low_engagement", "high", a, b,
                "Consider an interactive activity (quick poll, pair discussion or question round).",
                f"Class engagement stayed low for {(b - a + 1) * win:.0f}s.",
                {"mean_engagement": round(m, 1), "windows": b - a + 1})

    # 2. Attention drop (rolling mean of 2 windows)
    s = pd.Series(eng, dtype="float").rolling(2, min_periods=1).mean()
    for i in range(2, len(s)):
        if s[i - 2] == s[i - 2] and s[i] == s[i] and s[i - 2] - s[i] >= cfg.attention_drop_points:
            add("attention_drop", "medium", i - 1, i,
                "Attention dropped noticeably; a short break, recap or change of pace may help.",
                f"Engagement fell by {s[i - 2] - s[i]:.0f} points within {2 * win:.0f}s.",
                {"before": round(float(s[i - 2]), 1), "after": round(float(s[i]), 1)})
            break  # report the first clear drop only; avoid repeating

    # 3. Peak engagement -> highlight effective segment
    valid = [(e, i) for i, e in enumerate(eng) if e == e]
    if valid:
        peak, pi = max(valid)
        if peak >= cfg.high_engagement:
            add("engagement_peak", "info", pi, pi,
                "Engagement peaked here; note what was happening and consider reusing that approach.",
                f"Highest class engagement of the session ({peak:.0f}).",
                {"engagement": peak})

    # 4. Noisy conditions -> classroom management cue
    if "audio_activity" in tl:
        noisy = [a == "noisy" for a in tl["audio_activity"]]
        for a, b in _runs(noisy):
            if b - a + 1 >= cfg.noisy_windows:
                add("high_noise", "medium", a, b,
                    "Noise stayed high; a classroom-management cue (attention signal, quieter group work) may help.",
                    f"Ambient noise at or above {cfg.noisy_db:.0f} dBFS for {(b - a + 1) * win:.0f}s.",
                    {"mean_noise_db": round(float(tl.loc[a:b, 'noise_db'].mean()), 1)})

        # 5. Long stretch with little speech -> invite interaction
        quiet = [(sr == sr and sr < 0.05) for sr in tl["speech_ratio"]]
        for a, b in _runs(quiet):
            if b - a + 1 >= cfg.low_speech_windows:
                add("low_interaction", "low", a, b,
                    "Little spoken interaction for a long stretch; consider inviting questions or discussion.",
                    f"Almost no speech detected for {(b - a + 1) * win:.0f}s.", {})

    if not out and valid:
        mean = sum(e for e, _ in valid) / len(valid)
        out.append({"type": "steady", "priority": "info", "from_s": 0.0,
                    "to_s": float(tl["t_end"].iloc[-1]), "time_range": span(0, len(tl) - 1),
                    "suggestion": "No notable issues detected; engagement looks steady.",
                    "why": f"Mean class engagement {mean:.0f} with no sustained dips.",
                    "evidence": {"mean_engagement": round(mean, 1)}})
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    return sorted(out, key=lambda x: (order[x["priority"]], x["from_s"]))
