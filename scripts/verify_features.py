"""Verify the key features and print PASS/FAIL for each. Run: python scripts/verify_features.py"""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from attendance_agent.config import Config
from attendance_agent.vision import FaceTracker, compute_attendance
from attendance_agent.vision.detector import Detection
from attendance_agent.audio import AudioAnalyzer
from attendance_agent.agent import EngagementScorer
from attendance_agent.analytics import generate_suggestions
from attendance_agent.demo import build_demo
from attendance_agent.pipeline import SessionPipeline

results = []
def check(name, ok, detail=""):
    ok = bool(ok); results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))

# 1. Presence-based attendance: 1 s blip is absent, 40 s is present
cfg = Config(min_track_hits=1, min_presence_seconds=30)
tr = FaceTracker(cfg)
for i in range(5): tr.update(i * .2, [Detection((50, 50, 80, 100), "frontal")])
for i in range(200): tr.update(i * .2, [Detection((400, 300, 80, 100), "frontal")])
att = compute_attendance(tr.finalize(), .2, 60, cfg)
check("Attendance uses cumulative presence, not one frame",
      [(a["presence_seconds"], a["status"]) for a in att] == [(1.0, "absent"), (40.0, "present")],
      str([(a["participant_id"], a["presence_seconds"], a["status"]) for a in att]))

# 2. Tracker keeps one ID through a brief occlusion
tr = FaceTracker(Config(min_track_hits=2))
for i in range(10): tr.update(i * .2, [Detection((100 + i, 100, 80, 100), "frontal")])
tr.update(2.0, []); tr.update(2.6, [Detection((112, 100, 80, 100), "frontal")])
ids = {o.track_id for o in tr.finalize()}
check("Tracking survives a brief occlusion (single ID)", ids == {1}, f"ids={ids}")

# 3. Audio: silence / speech / noise
sr = 16000; rng = np.random.default_rng(0); t = np.arange(sr * 30) / sr
y = rng.normal(0, .002, sr * 30)
v = (np.sin(2*np.pi*150*t) + .5*np.sin(2*np.pi*300*t)) * np.clip(np.sin(2*np.pi*3*t), 0, None)
y[10*sr:20*sr] += .1 * v[10*sr:20*sr]; y[20*sr:] += rng.normal(0, .08, 10*sr)
a = AudioAnalyzer(Config()).analyze(y.astype("float32"), 30)
check("Audio windows classified silent / speech / noisy",
      list(a["audio_activity"]) == ["silent", "normal_discussion", "noisy"], str(list(a["audio_activity"])))

# 4. Scoring is explainable and ordered
s = EngagementScorer(Config())
good = dict(presence_fraction=1, facing_ratio=1, movement=.6, inactivity_ratio=.1)
bad = dict(presence_fraction=1, facing_ratio=.1, movement=0, inactivity_ratio=1)
g = s.score(good, dict(noise_db=-50, speech_ratio=.4)); b = s.score(bad, dict(noise_db=-20, speech_ratio=0))
check("Engaged score > disengaged score, with reasons", g[0] > 85 and b[0] < 30 and b[2], f"{g[0]} vs {b[0]}; {b[2]}")
check("Falls back to vision-only when audio is missing", "audio" not in s.score(good, None)[1])

# 5. Suggestions
v = [85, 85, 80, 30, 25, 20, 30, 80]; n = len(v)
tl = pd.DataFrame({"window": range(n), "t_start": [i*10. for i in range(n)], "t_end": [(i+1)*10. for i in range(n)],
    "class_engagement": v, "noise_db": [-50]*3 + [-20]*3 + [-50]*2, "speech_ratio": [.3]*n,
    "audio_activity": ["normal_discussion"]*3 + ["noisy"]*3 + ["normal_discussion"]*2})
kinds = {x["type"] for x in generate_suggestions(tl, Config())}
check("Teaching suggestions generated from trends",
      {"sustained_low_engagement", "attention_drop", "high_noise", "engagement_peak"} <= kinds, str(sorted(kinds)))

# 6. Full pipeline + outputs + student analytics privacy
cfg = Config(); pipe = SessionPipeline(cfg)
obs, dt, dur, audio = build_demo(cfg)
res = pipe.analyse(obs, dt, dur, audio)
status = {x["participant_id"]: x["status"] for x in res.attendance}
check("Demo session: 5 detected, 4 present, brief visitor (P4) absent",
      len(status) == 5 and sum(v == "present" for v in status.values()) == 4 and status["P4"] == "absent", str(status))
out = pipe.save(res, Path(tempfile.mkdtemp()) / "s")
need = {"attendance.csv", "participant_engagement.csv", "class_timeline.csv", "audio_metrics.csv",
        "suggestions.json", "student_analytics.json", "summary.json"}
check("All output files written", need <= {p.name for p in out.iterdir()})
blob = json.dumps(res.student_analytics).lower()
check("Student analytics contain no rankings/leaderboard", not any(w in blob for w in ("rank", "leaderboard", "percentile")))

# 7. Offline / privacy
src = "".join(p.read_text() for p in Path(__file__).resolve().parents[1].glob("attendance_agent/**/*.py"))
check("No network code in the package", not any(w in src for w in ("import requests", "urllib", "http.client", "import socket")))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
