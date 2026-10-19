"""Activity 2.1 - Face detection.

Uses OpenCV's bundled Haar cascades, so no model download or network access
is needed. A frontal detection means the person is facing the camera; a
profile detection (left or right) is used as a head-pose proxy meaning the
person has turned away. No face images or embeddings are stored.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Config


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h in original-frame pixels
    pose: str                        # "frontal" | "profile"


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


class FaceDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        base = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self.frontal.empty():
            raise RuntimeError("Could not load OpenCV frontal-face cascade")
        if cfg.detect_profile and self.profile.empty():
            raise RuntimeError("Could not load OpenCV profile-face cascade")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.cfg.detect_max_width / float(w))
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)          # robustness to variable lighting
        m = self.cfg.min_face_size
        found: list[Detection] = []

        for (x, y, fw, fh) in self.frontal.detectMultiScale(gray, 1.1, 5, minSize=(m, m)):
            found.append(Detection((int(x), int(y), int(fw), int(fh)), "frontal"))

        if self.cfg.detect_profile:
            sw = gray.shape[1]
            for flipped in (False, True):
                g = cv2.flip(gray, 1) if flipped else gray
                for (x, y, fw, fh) in self.profile.detectMultiScale(g, 1.1, 5, minSize=(m, m)):
                    if flipped:
                        x = sw - x - fw
                    box = (int(x), int(y), int(fw), int(fh))
                    if all(_iou(box, d.bbox) < 0.3 for d in found):
                        found.append(Detection(box, "profile"))

        inv = 1.0 / scale
        return [Detection(tuple(int(round(v * inv)) for v in d.bbox), d.pose) for d in found]
