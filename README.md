# Multimodal Attendance & Engagement Agent

An AI-powered, **offline and privacy-preserving** classroom intelligence system. It combines
computer vision, audio signal processing and rule-based agent reasoning to:

- mark **attendance** from sustained presence (never a single frame),
- track **engagement** over time by fusing head-pose, movement, inactivity and audio signals,
- give instructors **context-aware teaching suggestions**,
- give learners a **personal session summary** (no grades, rankings or peer comparison).

It is decision support: it suggests, humans decide.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m attendance_agent --demo                      # synthetic session, no camera needed
python -m attendance_agent --video classroom.mp4       # recorded class (audio read from the video)
python -m attendance_agent --camera 0 --max-seconds 600 --show   # live webcam + microphone
streamlit run dashboard/app.py                         # optional dashboard
pytest -q                                              # run tests
```

Recorded-video audio extraction needs `ffmpeg` on your PATH; live microphone capture needs PortAudio
(via `sounddevice`). If either is missing the agent **falls back to vision-only scoring** and says so.
Pass `--audio file.wav` to supply audio separately, `--no-audio` to skip it, and
`--config config.example.json` to override thresholds.

Each run writes `output/session_<timestamp>/`:
`attendance.csv`, `participant_engagement.csv`, `class_timeline.csv`, `audio_metrics.csv`,
`suggestions.json`, `student_analytics.json`, `summary.json`, `config_used.json`.

## Milestone map

| Milestone | Activity | Where |
|---|---|---|
| 1. Requirements & environment | 1.1 `requirements.txt`, 1.2 venv setup | `requirements.txt`, Quick start |
| 2. Vision module | 2.1 face detection & tracking | `attendance_agent/vision/detector.py`, `tracker.py` |
| | 2.2 presence-based attendance | `vision/attendance.py` |
| 3. Audio module | 3.1 signal extraction | `audio/extraction.py` |
| | 3.2 noise & speech analysis | `audio/analysis.py` |
| 4. Multimodal agent | 4.1/4.2 fusion | `agent/fusion.py` |
| | 4.3 engagement scoring | `agent/scoring.py` |
| 5. Analytics & recommendations | 5.1 teaching suggestions | `analytics/suggestions.py` |
| | 5.2 student analytics | `analytics/student_analytics.py` |
| 6. Interface, testing, deployment | 6.1 core app | `attendance_agent/main.py`, `pipeline.py` |
| | 6.2 Streamlit dashboard | `dashboard/app.py` |
| | 6.3 validation & edge cases | `tests/` |

## How it works

```
video ──► FaceDetector ──► FaceTracker ──► attendance (cumulative presence ≥ threshold)
 (Haar frontal+profile)   (anonymous IDs)   └─► window features: presence, facing, movement, inactivity ─┐
                                                                                                         ├─► fusion ─► engagement score (0-100, per person per window)
audio ──► mono 16 kHz ──► frame energy / ZCR / flux ──► per-window noise dB, speech ratio, activity ─────┘        │
                                                                                                                  ├─► instructor suggestions
                                                                                                                  └─► student session analytics
```

**Engagement score** = weighted sum of presence (0.15), attention/facing (0.40), activity (0.20) and audio
(0.25). Without audio the weights are renormalised over the visual components. Every score stores its
components and plain-language reasons (e.g. "noisy room", "often looking away"). All thresholds are in
`attendance_agent/config.py` and can be overridden by JSON.

**Suggestion rules:** sustained low engagement → interactive activity; sharp drop → break/recap; peak →
note what worked; sustained high noise → classroom-management cue; long silence → invite discussion.

## Privacy by design

- Fully offline: no network calls, no cloud APIs.
- No face recognition, embeddings or face images are stored. Participants are anonymous IDs (`P1`, `P2`…).
- Raw audio is analysed in memory; only aggregate metrics are saved.
- Student analytics contain only that learner's own data: no rankings, leaderboards or comparisons.

## Validation & known limitations (Activity 6.3)

Covered by automated tests: brief detections not counted as attendance, intermittent presence
accumulating, occlusion bridging, false-positive track filtering, missing/short audio, no audio track,
empty sessions, missing video file, noisy/silent/speech audio windows, and the full recorded-video path.

Be aware of these honest limits:

- **Head pose is a proxy.** Haar cascades distinguish frontal vs side-on faces only; they cannot measure
  gaze or detect looking down. Treat "attention" as "facing forward".
- **No identity.** Because the system never recognises people, someone who leaves for longer than
  `occlusion_tolerance_s` returns as a new ID (and their presence accumulates under both). Increase the
  tolerance for classrooms where people are often hidden.
- **Haar detectors miss** small, masked or poorly lit faces and occasionally give false positives; the
  `min_track_hits` and presence threshold filter most of these. Tune `min_face_size` and `process_fps`.
- **Audio levels are relative to your microphone gain** (dBFS). Calibrate `noisy_db` / `quiet_db` per room.
- **Speech detection is energy-based**, so it detects sound activity, not who is speaking or what is said.
- Defaults are reasonable starting points, not validated cut-offs. Pilot them on your own sessions.
