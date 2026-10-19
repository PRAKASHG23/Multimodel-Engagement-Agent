"""Activity 3.2 - Noise and speech activity analysis.

For each short frame we compute energy (dBFS), zero-crossing rate and spectral
flux. Frames well above the adaptive noise floor with speech-like
characteristics are counted as speech. Results are aggregated per window so
they line up with the visual features.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import Config


class AudioAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # -- frame level ----------------------------------------------------
    def _frame_features(self, y: np.ndarray):
        sr = self.cfg.sample_rate
        n = int(sr * self.cfg.audio_frame_ms / 1000)
        hop = int(sr * self.cfg.audio_hop_ms / 1000)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        idx = np.arange(0, len(y) - n + 1, hop)
        frames = np.stack([y[i:i + n] for i in idx]) * np.hanning(n)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        signs = np.signbit(frames).astype(np.int8)
        zcr = np.abs(np.diff(signs, axis=1)).mean(axis=1)
        mag = np.abs(np.fft.rfft(frames, axis=1))
        flux = np.r_[0.0, np.sqrt((np.diff(mag, axis=0) ** 2).sum(axis=1))]
        flux = flux / (mag.sum(axis=1) + 1e-9)
        times = idx / sr
        return times, db, zcr, flux

    # -- window level -----------------------------------------------------
    def analyze(self, y: np.ndarray | None, duration: float) -> pd.DataFrame | None:
        """Return one row per window, or None when no audio is available."""
        if y is None or len(y) < self.cfg.sample_rate * 0.5:
            return None
        cfg = self.cfg
        times, db, zcr, flux = self._frame_features(y)
        floor = float(np.percentile(db, 10))                 # adaptive session noise floor
        speech = (db > floor + cfg.speech_margin_db) & (zcr > 0.01) & (zcr < 0.45)

        win = cfg.window_seconds
        n_win = max(1, math.ceil(duration / win))
        rows = []
        for w in range(n_win):
            t0, t1 = w * win, (w + 1) * win
            m = (times >= t0) & (times < t1)
            if m.sum() < 5:      # missing audio segment
                rows.append({"window": w, "t_start": t0, "t_end": min(t1, duration),
                             "noise_db": np.nan, "speech_ratio": np.nan,
                             "amplitude_var_db": np.nan, "audio_activity": "no_audio"})
                continue
            sp = speech[m]
            non_speech_db = db[m][~sp]
            ambient = float(np.median(non_speech_db)) if len(non_speech_db) else float(np.median(db[m]))
            ratio = float(sp.mean())
            rows.append({
                "window": w, "t_start": t0, "t_end": min(t1, duration),
                "noise_db": round(ambient, 1),
                "speech_ratio": round(ratio, 3),
                "amplitude_var_db": round(float(db[m].std()), 2),
                "audio_activity": self._label(ambient, ratio),
            })
        return pd.DataFrame(rows)

    def _label(self, ambient_db: float, speech_ratio: float) -> str:
        c = self.cfg
        if ambient_db >= c.noisy_db:
            return "noisy"
        if speech_ratio < 0.05:
            return "silent" if ambient_db <= c.quiet_db else "quiet"
        if speech_ratio >= 0.55:
            return "active_discussion"
        return "normal_discussion"

    @staticmethod
    def session_metrics(df: pd.DataFrame | None) -> dict:
        if df is None or df.empty:
            return {"audio_available": False}
        valid = df.dropna(subset=["noise_db"])
        return {
            "audio_available": True,
            "mean_noise_db": round(float(valid["noise_db"].mean()), 1),
            "max_noise_db": round(float(valid["noise_db"].max()), 1),
            "mean_speech_ratio": round(float(valid["speech_ratio"].mean()), 3),
            "noisy_windows": int((valid["audio_activity"] == "noisy").sum()),
            "silent_windows": int((valid["audio_activity"] == "silent").sum()),
        }
