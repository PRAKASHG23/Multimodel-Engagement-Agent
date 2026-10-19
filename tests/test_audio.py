import numpy as np
from attendance_agent.config import Config
from attendance_agent.audio import AudioAnalyzer


def make(sr=16000, secs=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(sr * secs) / sr
    y = rng.normal(0, 0.002, sr * secs)
    voice = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
    env = np.clip(np.sin(2 * np.pi * 3 * t), 0, None)
    y[10 * sr:20 * sr] += 0.1 * (voice * env)[10 * sr:20 * sr]   # speech in window 1
    seg = y[20 * sr:30 * sr]
    seg += rng.normal(0, 0.08, len(seg))                         # noise in window 2
    return y.astype(np.float32)


def test_windows_distinguish_silence_speech_noise():
    cfg = Config()
    df = AudioAnalyzer(cfg).analyze(make(), 30)
    assert list(df["audio_activity"][:1]) in (["silent"], ["quiet"])
    assert df.loc[1, "speech_ratio"] > 0.2
    assert df.loc[2, "audio_activity"] == "noisy"
    assert df.loc[2, "noise_db"] > df.loc[0, "noise_db"] + 20


def test_missing_audio_returns_none_and_short_audio_ok():
    a = AudioAnalyzer(Config())
    assert a.analyze(None, 10) is None
    assert a.analyze(np.zeros(100, dtype=np.float32), 10) is None
    assert AudioAnalyzer.session_metrics(None) == {"audio_available": False}


def test_audio_shorter_than_session_marks_missing_windows():
    df = AudioAnalyzer(Config()).analyze(make(secs=12), 30)
    assert df.loc[2, "audio_activity"] == "no_audio"
