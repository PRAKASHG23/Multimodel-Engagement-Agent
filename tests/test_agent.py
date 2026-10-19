import math
import pandas as pd
from attendance_agent.config import Config
from attendance_agent.agent import EngagementScorer, MultimodalAgent
from attendance_agent.analytics import generate_suggestions, generate_student_analytics

VIS_GOOD = dict(presence_fraction=1.0, facing_ratio=1.0, movement=0.6, inactivity_ratio=0.1)
VIS_BAD = dict(presence_fraction=1.0, facing_ratio=0.1, movement=0.0, inactivity_ratio=1.0)
AUD_GOOD = dict(noise_db=-50, speech_ratio=0.4)
AUD_NOISY = dict(noise_db=-20, speech_ratio=0.0)


def test_scores_ordering_and_reasons():
    s = EngagementScorer(Config())
    good, _, r1 = s.score(VIS_GOOD, AUD_GOOD)
    bad, comp, r2 = s.score(VIS_BAD, AUD_NOISY)
    assert good > 85 and bad < 30
    assert "noisy room" in r2 and "often looking away from the front" in r2
    assert set(comp) == {"presence", "attention", "activity", "audio"}


def test_visual_only_fallback_when_audio_missing():
    s = EngagementScorer(Config())
    score, comp, _ = s.score(VIS_GOOD, None)
    assert "audio" not in comp and score > 80
    score2, comp2, _ = s.score(VIS_GOOD, dict(noise_db=float("nan"), speech_ratio=float("nan")))
    assert "audio" not in comp2


def test_low_presence_window_not_scored():
    s = EngagementScorer(Config())
    assert s.score({**VIS_GOOD, "presence_fraction": 0.1}, None) is None


def test_fidgeting_penalised():
    s = EngagementScorer(Config())
    calm, _, _ = s.score(VIS_GOOD, None)
    fid, _, _ = s.score({**VIS_GOOD, "movement": 8.0}, None)
    assert fid < calm


def _timeline(vals, noise=None):
    n = len(vals)
    return pd.DataFrame({
        "window": range(n), "t_start": [i * 10.0 for i in range(n)],
        "t_end": [(i + 1) * 10.0 for i in range(n)],
        "class_engagement": vals,
        "noise_db": noise or [-50.0] * n,
        "speech_ratio": [0.3] * n,
        "audio_activity": ["noisy" if (noise and x > -30) else "normal_discussion" for x in (noise or [-50] * n)],
    })


def test_suggestions_sustained_low_drop_peak_noise():
    cfg = Config()
    tl = _timeline([85, 85, 80, 30, 25, 20, 30, 80], noise=[-50] * 3 + [-20] * 3 + [-50] * 2)
    kinds = {s["type"] for s in generate_suggestions(tl, cfg)}
    assert {"sustained_low_engagement", "attention_drop", "engagement_peak", "high_noise"} <= kinds


def test_suggestions_steady_session_and_nan_tolerance():
    cfg = Config()
    out = generate_suggestions(_timeline([60, 62, float("nan"), 61, 60]), cfg)
    assert [s["type"] for s in out] == ["steady"]
    assert generate_suggestions(pd.DataFrame(), cfg) == []


def test_student_analytics_no_ranking_fields():
    cfg = Config()
    att = [{"participant_id": "P1", "status": "present", "presence_seconds": 100.0,
            "presence_percent": 90.0, "first_seen": "0", "last_seen": "100"},
           {"participant_id": "P2", "status": "absent", "presence_seconds": 3.0,
            "presence_percent": 3.0, "first_seen": "0", "last_seen": "3"}]
    ps = pd.DataFrame({"participant_id": ["P1"] * 6, "window": range(6),
                       "t_start": [i * 10.0 for i in range(6)],
                       "engagement": [30, 35, 70, 80, 85, 90]})
    out = generate_student_analytics(att, ps, cfg)
    p1, p2 = out
    assert p1["engagement"]["average"] > 60 and "focus built up" in " ".join(p1["notes"])
    assert p2["engagement"] is None
    blob = str(out).lower()
    assert "rank" not in blob and "leaderboard" not in blob and "percentile" not in blob
