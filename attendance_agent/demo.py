"""Synthetic classroom session (no camera, no real people, no files needed).

Feeds scripted face observations and a generated audio waveform through the
real analysis stack. Used for the demo dashboard and for testing.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .vision.tracker import Observation


def synth_audio(duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Quiet room + speech-like bursts; a noisy stretch at 240-300s."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    y = rng.normal(0, 0.002, n)                                   # ~ -54 dBFS room tone
    # speech-like: harmonic voice with syllable-rate amplitude modulation
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), 1))
    env = np.clip(np.sin(2 * np.pi * 3.5 * t), 0, None)           # syllables
    talking = np.zeros(n)
    for start, end in ((10, 60), (80, 150), (170, 230), (320, 400), (430, 570)):
        if start < duration:
            talking[int(start * sr):int(min(end, duration) * sr)] = 1
    y += 0.08 * voice * env * talking
    if duration > 300:                                            # noisy segment
        a, b = int(240 * sr), int(300 * sr)
        y[a:b] += rng.normal(0, 0.06, b - a)
    return np.clip(y, -1, 1).astype(np.float32)


def synth_observations(duration: float, fps: float, rng: np.random.Generator) -> list[Observation]:
    obs: list[Observation] = []
    dt = 1.0 / fps
    seats = {1: (160, 200), 2: (320, 200), 3: (480, 200), 4: (160, 360), 5: (480, 360)}
    for i in range(int(duration * fps)):
        t = i * dt
        for pid, (x, y) in seats.items():
            # P3 arrives 90s late, P4 only passes through for 6s (should stay "absent")
            if pid == 3 and t < 90:
                continue
            if pid == 4 and not (200 <= t < 206):
                continue
            facing = 0.95
            if pid == 2 and t > 200:
                facing = 0.25                                     # P2 drifts away
            if 240 <= t < 300:
                facing -= 0.35                                    # class-wide dip during noise
            pose = "frontal" if rng.random() < facing else "profile"
            if pid == 2 and 120 <= t < 124:                       # brief occlusion
                continue
            wiggle = 6.0 if pid == 5 else 1.0                     # P5 is restless
            cx = x + rng.normal(0, wiggle) + (3 * np.sin(t / 7) if pid != 1 else 0)
            cy = y + rng.normal(0, wiggle)
            obs.append(Observation(t, pid, cx, cy, 90, 110, pose))
    return obs


def build_demo(cfg: Config | None = None, duration: float = 600.0, seed: int = 7):
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    fps = cfg.process_fps
    return synth_observations(duration, fps, rng), 1.0 / fps, duration, synth_audio(duration, cfg.sample_rate, rng)
