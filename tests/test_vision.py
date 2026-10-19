from attendance_agent.config import Config
from attendance_agent.vision import FaceTracker, compute_attendance, extract_window_features
from attendance_agent.vision.detector import Detection


def det(x, y, pose="frontal"):
    return Detection((x, y, 80, 100), pose)


def test_tracker_keeps_identity_and_survives_brief_occlusion():
    cfg = Config(min_track_hits=2)
    tr = FaceTracker(cfg)
    for i in range(10):
        tr.update(i * 0.2, [det(100 + i, 100)])
    tr.update(2.0, [])                       # occluded
    tr.update(2.6, [det(112, 100)])          # back within tolerance
    ids = {o.track_id for o in tr.finalize()}
    assert ids == {1}


def test_tracker_new_id_after_long_absence_and_noise_filtered():
    cfg = Config(min_track_hits=3, occlusion_tolerance_s=1.0)
    tr = FaceTracker(cfg)
    for i in range(10):
        tr.update(i * 0.2, [det(100, 100)])
    tr.update(2.0, [det(500, 300)])          # one-frame false positive
    for i in range(10):
        tr.update(10 + i * 0.2, [det(100, 100)])   # returns after >1s
    ids = {o.track_id for o in tr.finalize()}
    assert ids == {1, 2}                      # false positive dropped, returnee relabelled


def test_two_people_get_separate_ids():
    tr = FaceTracker(Config(min_track_hits=2))
    for i in range(10):
        tr.update(i * 0.2, [det(100, 100), det(400, 100)])
    assert {o.track_id for o in tr.finalize()} == {1, 2}


def test_attendance_threshold_not_single_frame():
    cfg = Config(min_presence_seconds=30, min_track_hits=1)
    tr = FaceTracker(cfg)
    for i in range(5):                        # 1 second blip
        tr.update(i * 0.2, [det(50, 50)])
    for i in range(200):                      # 40 seconds of a second person
        tr.update(i * 0.2, [det(400, 300)])
    obs = tr.finalize()
    rec = {r["participant_id"]: r for r in compute_attendance(obs, 0.2, 60, cfg)}
    statuses = sorted(r["status"] for r in rec.values())
    assert statuses == ["absent", "present"]


def test_intermittent_presence_accumulates():
    cfg = Config(min_presence_seconds=10, min_track_hits=1, occlusion_tolerance_s=0.5)
    tr = FaceTracker(cfg)
    t = 0.0
    for _ in range(4):                        # 4 bursts of 3s, separated by long gaps
        for i in range(15):
            tr.update(t, [det(100, 100)]); t += 0.2
        t += 20
    rec = compute_attendance(tr.finalize(), 0.2, 100, cfg)
    assert sum(r["presence_seconds"] for r in rec) >= 10


def test_window_features_detect_stillness_and_turning():
    cfg = Config(min_track_hits=1)
    tr = FaceTracker(cfg)
    for i in range(50):                       # still, frontal, 10 s
        tr.update(i * 0.2, [det(100, 100)])
    rows = extract_window_features(tr.finalize(), 0.2, 10, cfg)
    r = rows[0]
    assert r["facing_ratio"] == 1.0 and r["inactivity_ratio"] > 0.9
    assert r["presence_fraction"] > 0.95
