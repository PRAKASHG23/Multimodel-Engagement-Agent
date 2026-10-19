"""Central, adjustable configuration.

Every threshold used by the rule-based logic lives here so instructors can
tune the system to their classroom without touching code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # ---- Vision / tracking (Milestone 2) ----
    process_fps: float = 5.0            # frames analysed per second of video
    detect_max_width: int = 640         # frames are downscaled for detection
    min_face_size: int = 30             # px, on the downscaled frame
    detect_profile: bool = True         # also detect side-on faces (head-turn proxy)
    track_iou_match: float = 0.15       # min IoU to associate a detection to a track
    track_dist_factor: float = 0.75     # max centre distance, in face-widths
    track_smoothing: float = 0.5        # EMA on bbox (reduces detector jitter)
    occlusion_tolerance_s: float = 2.0  # bridge brief occlusions up to this long
    min_track_hits: int = 3             # tracks seen fewer times are discarded as noise

    # ---- Presence-based attendance (Activity 2.2) ----
    min_presence_seconds: float = 30.0  # cumulative presence needed to be "present"

    # ---- Audio (Milestone 3) ----
    sample_rate: int = 16000
    audio_frame_ms: int = 25
    audio_hop_ms: int = 10
    speech_margin_db: float = 10.0      # frame this far above noise floor => speech candidate
    noisy_db: float = -30.0             # ambient level (dBFS) regarded as "noisy"
    quiet_db: float = -55.0             # ambient level regarded as "quiet"

    # ---- Fusion & scoring (Milestone 4) ----
    window_seconds: float = 10.0        # engagement is scored per window
    min_window_presence: float = 0.3    # participants below this are not scored in a window
    still_speed: float = 0.25           # face-widths/s below which a person is "inactive"
    fidget_speed: float = 2.0           # face-widths/s above which movement is excessive
    target_speech_ratio: float = 0.30   # speech share regarded as healthy interaction
    weights_with_audio: dict = field(default_factory=lambda: {
        "presence": 0.15, "attention": 0.40, "activity": 0.20, "audio": 0.25})
    weights_visual_only: dict = field(default_factory=lambda: {
        "presence": 0.20, "attention": 0.55, "activity": 0.25})
    low_engagement: float = 40.0        # score < this => "low"
    high_engagement: float = 70.0       # score >= this => "high"

    # ---- Suggestions (Milestone 5) ----
    sustained_low_windows: int = 3      # consecutive low windows => suggestion
    attention_drop_points: float = 20.0 # drop between neighbouring rolling means
    noisy_windows: int = 2              # consecutive noisy windows => flag
    low_speech_windows: int = 6         # long stretch with little speech => flag

    # ---- Student analytics ----
    focus_threshold: float = 70.0
    dip_threshold: float = 40.0

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text())
            valid = {f.name for f in fields(cls)}
            unknown = set(data) - valid
            if unknown:
                raise ValueError(f"Unknown config keys: {sorted(unknown)}")
            for k, v in data.items():
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
