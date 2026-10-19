"""End-to-end pipeline: video + audio -> attendance, engagement, suggestions, analytics."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .agent import MultimodalAgent
from .analytics import generate_student_analytics, generate_suggestions
from .audio import AudioAnalyzer, LiveAudioRecorder, extract_audio_from_video, load_audio
from .config import Config
from .vision import FaceTracker, compute_attendance, extract_window_features
from .vision.tracker import Observation


@dataclass
class SessionResult:
    attendance: list[dict]
    participant_scores: pd.DataFrame
    class_timeline: pd.DataFrame
    audio_windows: pd.DataFrame | None
    suggestions: list[dict]
    student_analytics: list[dict]
    summary: dict


class SessionPipeline:
    def __init__(self, cfg: Config | None = None, detector=None):
        self.cfg = cfg or Config()
        self._detector = detector          # injectable (tests / alternative detectors)

    @property
    def detector(self):
        if self._detector is None:
            from .vision import FaceDetector
            self._detector = FaceDetector(self.cfg)
        return self._detector

    # ------------------------------------------------------------------
    # Analysis stage (shared by live runs, recorded runs, demo and tests)
    # ------------------------------------------------------------------
    def analyse(self, observations: list[Observation], sample_dt: float, duration: float,
                audio: np.ndarray | None = None, session_start: datetime | None = None) -> SessionResult:
        cfg = self.cfg
        session_start = session_start or datetime.now()
        duration = max(duration, sample_dt)

        attendance = compute_attendance(observations, sample_dt, duration, cfg, session_start)
        visual = extract_window_features(observations, sample_dt, duration, cfg)
        audio_df = AudioAnalyzer(cfg).analyze(audio, duration)
        scores, timeline = MultimodalAgent(cfg).fuse(visual, audio_df, duration)
        suggestions = generate_suggestions(timeline, cfg)
        students = generate_student_analytics(attendance, scores, cfg)

        valid = timeline["class_engagement"].dropna()
        summary = {
            "session_start": session_start.isoformat(timespec="seconds"),
            "duration_seconds": round(duration, 1),
            "participants_detected": len(attendance),
            "participants_present": sum(a["status"] == "present" for a in attendance),
            "mean_class_engagement": round(float(valid.mean()), 1) if len(valid) else None,
            **AudioAnalyzer.session_metrics(audio_df),
            "privacy": "Processed offline; no face images, embeddings or audio recordings are stored.",
        }
        return SessionResult(attendance, scores, timeline, audio_df, suggestions, students, summary)

    # ------------------------------------------------------------------
    # Capture stage (Activity 2.1 + 3.1)
    # ------------------------------------------------------------------
    def run_video(self, source: str | int, audio_path: str | None = None, use_audio: bool = True,
                  show: bool = False, max_seconds: float | None = None) -> SessionResult:
        cfg = self.cfg
        live = isinstance(source, int)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if live or fps < 1 or fps > 240:
            fps = 30.0 if not live else fps or 30.0
        step = max(1, round(fps / cfg.process_fps))
        tracker = FaceTracker(cfg)

        recorder = None
        if live and use_audio:
            recorder = LiveAudioRecorder(cfg.sample_rate)
            recorder.start()

        start_wall, session_start = time.monotonic(), datetime.now()
        idx, last_t = 0, 0.0
        try:
            while True:
                if not cap.grab():
                    break                       # end of file / camera lost
                t = (time.monotonic() - start_wall) if live else idx / fps
                if max_seconds and t > max_seconds:
                    break
                if idx % step == 0 or live and (t - last_t) >= 1.0 / cfg.process_fps:
                    ok, frame = cap.retrieve()
                    if ok:
                        dets = self.detector.detect(frame)
                        tracker.update(t, dets)
                        last_t = t
                        if show:
                            self._draw(frame, tracker, dets)
                            cv2.imshow("Attendance & Engagement (q to stop)", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                idx += 1
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()

        duration = (time.monotonic() - start_wall) if live else idx / fps
        audio = None
        if use_audio:
            if recorder is not None:
                audio = recorder.stop()
            elif audio_path:
                audio = load_audio(audio_path, cfg.sample_rate)
            elif not live:
                audio = extract_audio_from_video(source, cfg.sample_rate)
            if audio is None:
                print("[audio] no audio available; scoring from vision only")
        if audio is not None:
            audio = audio[: int(duration * cfg.sample_rate)] if duration > 0 else audio

        return self.analyse(tracker.finalize(), tracker.sample_interval(), duration, audio, session_start)

    @staticmethod
    def _draw(frame, tracker: FaceTracker, dets):
        recent = [o for o in tracker.observations[-len(dets):]] if dets else []
        for o in recent:
            x, y = int(o.cx - o.w / 2), int(o.cy - o.h / 2)
            cv2.rectangle(frame, (x, y), (int(x + o.w), int(y + o.h)), (0, 200, 0), 2)
            cv2.putText(frame, f"#{o.track_id} {o.pose}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    # ------------------------------------------------------------------
    # Output stage
    # ------------------------------------------------------------------
    def save(self, result: SessionResult, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.attendance).to_csv(out / "attendance.csv", index=False)
        ps = result.participant_scores.copy()
        if len(ps):
            ps["components"] = ps["components"].map(json.dumps)
        ps.to_csv(out / "participant_engagement.csv", index=False)
        result.class_timeline.to_csv(out / "class_timeline.csv", index=False)
        if result.audio_windows is not None:
            result.audio_windows.to_csv(out / "audio_metrics.csv", index=False)
        (out / "suggestions.json").write_text(json.dumps(result.suggestions, indent=2))
        (out / "student_analytics.json").write_text(json.dumps(result.student_analytics, indent=2))
        (out / "summary.json").write_text(json.dumps(result.summary, indent=2))
        (out / "config_used.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        return out
