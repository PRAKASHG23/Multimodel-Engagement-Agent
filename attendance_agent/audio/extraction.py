"""Activity 3.1 - Audio signal extraction.

Audio comes from a recorded file, the soundtrack of a recorded video (via a
local ``ffmpeg``), or a live microphone. Everything is normalised to a mono
waveform at a fixed sampling rate so all sources are processed identically.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from math import gcd


def _normalise(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:                       # channel standardisation -> mono
        y = y.mean(axis=1)
    if sr != target_sr:                  # sampling-rate normalisation
        g = gcd(int(sr), int(target_sr))
        y = resample_poly(y, target_sr // g, int(sr) // g).astype(np.float32)
    return y


def load_audio(path: str | Path, target_sr: int = 16000) -> np.ndarray | None:
    """Load a wav/flac/ogg file as a mono float32 waveform in [-1, 1]."""
    import soundfile as sf
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return None
    return _normalise(y, sr, target_sr)


def extract_audio_from_video(video_path: str | Path, target_sr: int = 16000) -> np.ndarray | None:
    """Pull the soundtrack out of a video. Returns None if there is none / no ffmpeg."""
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
               "-vn", "-ac", "1", "-ar", str(target_sr), "-f", "wav", str(wav)]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0 or not wav.exists() or wav.stat().st_size < 1000:
            return None
        return load_audio(wav, target_sr)


class LiveAudioRecorder:
    """Microphone capture running alongside the webcam (needs ``sounddevice``)."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self.available = False
        self.t0: float | None = None

    def start(self) -> bool:
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                          dtype="float32", callback=self._cb)
            self._stream.start()
            self.t0 = time.monotonic()
            self.available = True
        except Exception as exc:  # no mic / PortAudio missing
            print(f"[audio] live microphone unavailable ({exc}); continuing vision-only")
            self.available = False
        return self.available

    def _cb(self, indata, frames, t, status):
        self._chunks.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray | None:
        if self._stream is not None:
            self._stream.stop(); self._stream.close()
        if not self._chunks:
            return None
        return np.concatenate(self._chunks)
