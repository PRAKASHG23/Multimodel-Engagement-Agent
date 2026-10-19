import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from attendance_agent.config import Config
from attendance_agent.demo import build_demo
from attendance_agent.pipeline import SessionPipeline
from attendance_agent.vision.detector import Detection


def test_demo_end_to_end_and_outputs(tmp_path):
    cfg = Config()
    pipe = SessionPipeline(cfg)
    obs, dt, dur, audio = build_demo(cfg, duration=420)
    res = pipe.analyse(obs, dt, dur, audio)
    status = {a["participant_id"]: a["status"] for a in res.attendance}
    assert status["P4"] == "absent" and status["P1"] == "present"      # brief pass-through not counted
    assert res.summary["audio_available"] is True
    assert any(s["type"] == "high_noise" for s in res.suggestions)
    out = pipe.save(res, tmp_path / "s")
    for f in ("attendance.csv", "participant_engagement.csv", "class_timeline.csv",
              "audio_metrics.csv", "suggestions.json", "student_analytics.json", "summary.json"):
        assert (out / f).exists(), f
    json.loads((out / "summary.json").read_text())


def test_vision_only_when_no_audio():
    cfg = Config()
    obs, dt, dur, _ = build_demo(cfg, duration=120)
    res = SessionPipeline(cfg).analyse(obs, dt, dur, None)
    assert res.summary["audio_available"] is False
    assert res.class_timeline["class_engagement"].notna().any()


def test_empty_session_does_not_crash():
    res = SessionPipeline(Config()).analyse([], 0.2, 30, None)
    assert res.attendance == [] and res.suggestions == []


class SquareDetector:
    """Test double: 'detects' bright squares so no real faces are needed."""
    def detect(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [Detection(cv2.boundingRect(c), "frontal") for c in cnts if cv2.contourArea(c) > 400]


def _make_video(path: Path, secs=20, fps=10, with_audio=True):
    silent = path.with_suffix(".silent.avi")
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for i in range(secs * fps):
        f = np.zeros((240, 320, 3), np.uint8)
        cv2.rectangle(f, (60, 60), (120, 130), (255, 255, 255), -1)           # always there
        if i > 10 * fps:
            cv2.rectangle(f, (200, 100), (260, 170), (255, 255, 255), -1)     # arrives at 10s
        vw.write(f)
    vw.release()
    if not (with_audio and shutil.which("ffmpeg")):
        return silent
    sr = 16000
    t = np.arange(secs * sr) / sr
    wav = (0.1 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    import soundfile as sf
    sf.write(path.with_suffix(".wav"), wav, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(path.with_suffix(".wav")),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_recorded_video_with_audio_end_to_end(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    video = _make_video(tmp_path / "class.mp4")
    cfg = Config(min_presence_seconds=5, window_seconds=5)
    res = SessionPipeline(cfg, detector=SquareDetector()).run_video(str(video))
    assert len(res.attendance) == 2
    by = {a["participant_id"]: a for a in res.attendance}
    assert by["P1"]["presence_seconds"] > by["P2"]["presence_seconds"] > 5
    assert res.summary["audio_available"] is True
    assert 18 <= res.summary["duration_seconds"] <= 22


def test_recorded_video_without_audio_falls_back(tmp_path):
    video = _make_video(tmp_path / "silent.mp4", with_audio=False)
    res = SessionPipeline(Config(min_presence_seconds=5), detector=SquareDetector()).run_video(str(video))
    assert res.summary["audio_available"] is False and len(res.attendance) == 2


def test_missing_video_raises(tmp_path):
    with pytest.raises(RuntimeError):
        SessionPipeline(Config(), detector=SquareDetector()).run_video(str(tmp_path / "nope.mp4"))


def test_real_haar_detector_loads_and_runs_on_blank_frame():
    from attendance_agent.vision import FaceDetector
    assert FaceDetector(Config()).detect(np.zeros((480, 640, 3), np.uint8)) == []
