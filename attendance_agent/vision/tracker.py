"""Activity 2.1 - Face tracking.

A lightweight IoU + centre-distance tracker. Each face gets an anonymous
identifier that persists across frames, slight head movements, orientation
changes and brief occlusions. It does NOT recognise identities: if someone
leaves for longer than the occlusion tolerance they come back as a new ID.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from .detector import Detection, _iou


@dataclass
class Observation:
    t: float
    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    pose: str


@dataclass
class _Track:
    id: int
    bbox: tuple[float, float, float, float]
    last_t: float
    hits: int = 1


class FaceTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.observations: list[Observation] = []
        self.frame_times: list[float] = []

    def update(self, t: float, detections: list[Detection]) -> list[Observation]:
        """Associate detections with tracks at time ``t`` (seconds)."""
        self.frame_times.append(t)
        # expire tracks that have been missing too long
        for tid in [k for k, tr in self._tracks.items()
                    if t - tr.last_t > self.cfg.occlusion_tolerance_s]:
            del self._tracks[tid]

        pairs = []
        for di, d in enumerate(detections):
            dcx, dcy = d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2
            for tid, tr in self._tracks.items():
                iou = _iou(d.bbox, tr.bbox)
                tcx, tcy = tr.bbox[0] + tr.bbox[2] / 2, tr.bbox[1] + tr.bbox[3] / 2
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5 / max(tr.bbox[2], 1)
                if iou >= self.cfg.track_iou_match or dist <= self.cfg.track_dist_factor:
                    pairs.append((-iou, dist, di, tid))
        pairs.sort()

        used_d, used_t, out = set(), set(), []
        a = self.cfg.track_smoothing
        for _, _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di); used_t.add(tid)
            tr, d = self._tracks[tid], detections[di]
            tr.bbox = tuple(a * n + (1 - a) * o for n, o in zip(d.bbox, tr.bbox))
            tr.last_t = t
            tr.hits += 1
            out.append(self._record(t, tr, d.pose))

        for di, d in enumerate(detections):
            if di in used_d:
                continue
            tr = _Track(self._next_id, tuple(float(v) for v in d.bbox), t)
            self._tracks[tr.id] = tr
            self._next_id += 1
            out.append(self._record(t, tr, d.pose))
        return out

    def _record(self, t: float, tr: _Track, pose: str) -> Observation:
        x, y, w, h = tr.bbox
        obs = Observation(t, tr.id, x + w / 2, y + h / 2, w, h, pose)
        self.observations.append(obs)
        return obs

    # ------------------------------------------------------------------
    def finalize(self) -> list[Observation]:
        """Drop spurious tracks (too few hits) and relabel as P1, P2, ... by first appearance."""
        counts: dict[int, int] = {}
        for o in self.observations:
            counts[o.track_id] = counts.get(o.track_id, 0) + 1
        keep = [tid for tid, c in counts.items() if c >= self.cfg.min_track_hits]
        first = {}
        for o in self.observations:
            first.setdefault(o.track_id, o.t)
        remap = {tid: i + 1 for i, tid in enumerate(sorted(keep, key=lambda k: first[k]))}
        return [Observation(o.t, remap[o.track_id], o.cx, o.cy, o.w, o.h, o.pose)
                for o in self.observations if o.track_id in remap]

    def sample_interval(self) -> float:
        """Typical seconds between analysed frames."""
        ft = self.frame_times
        if len(ft) < 2:
            return 1.0 / self.cfg.process_fps
        gaps = sorted(b - a for a, b in zip(ft, ft[1:]) if b > a)
        return gaps[len(gaps) // 2] if gaps else 1.0 / self.cfg.process_fps
