"""Opt-in local student registration and face-based attendance."""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timedelta
import hashlib
import hmac
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Callable, Iterator

import cv2
import numpy as np

from .config import Config

_FACE_SIZE = (160, 160)
_DEFAULT_CONFIDENCE = 60.0
_PASSWORD_SALT_BYTES = 16
_PASSWORD_HASH_BYTES = 64
_PASSWORD_SCRYPT_N = 2**14
_PASSWORD_SCRYPT_R = 8
_PASSWORD_SCRYPT_P = 1
_MIN_PASSWORD_LENGTH = 12
_MAX_PASSWORD_LENGTH = 1024
_MAX_ENROLLMENT_LENGTH = 128
_INSTRUCTOR_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def _validate_enrollment_number(enrollment_number: str) -> str:
    if not enrollment_number.isprintable() and enrollment_number.strip():
        raise ValueError("Enrollment number contains unsupported characters.")
    normalized = enrollment_number.strip()
    if not normalized:
        raise ValueError("A registered enrollment number is required.")
    if len(normalized) > _MAX_ENROLLMENT_LENGTH:
        raise ValueError(f"Enrollment number must be at most {_MAX_ENROLLMENT_LENGTH} characters.")
    if not normalized.isprintable():
        raise ValueError("Enrollment number contains unsupported characters.")
    return normalized


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_PASSWORD_SCRYPT_N,
        r=_PASSWORD_SCRYPT_R,
        p=_PASSWORD_SCRYPT_P,
        dklen=_PASSWORD_HASH_BYTES,
    )


def _prepare_face(photo: bytes, cfg: Config) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(photo, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Could not load OpenCV frontal-face cascade.")
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(cfg.min_face_size, cfg.min_face_size)
    )
    if len(faces) != 1:
        raise ValueError("Use a clear, front-facing photo containing exactly one detectable face.")
    x, y, width, height = faces[0]
    face = cv2.equalizeHist(gray[y:y + height, x:x + width])
    return cv2.resize(face, _FACE_SIZE, interpolation=cv2.INTER_AREA)


class StudentRegistry:
    """SQLite-backed registry; stored photos and face templates remain on this machine."""

    def __init__(self, database_path: str | Path, cfg: Config | None = None):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or Config()
        self._frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self._frontal.empty():
            raise RuntimeError("Could not load OpenCV frontal-face cascade.")
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    institution TEXT NOT NULL,
                    enrollment_number TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    photo BLOB NOT NULL,
                    face_template BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attendance_sessions (
                    id INTEGER PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    required_presence_seconds REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attendance_records (
                    session_id INTEGER NOT NULL REFERENCES attendance_sessions(id),
                    student_id INTEGER NOT NULL REFERENCES students(id),
                    presence_seconds REAL NOT NULL,
                    first_seen TEXT,
                    last_seen TEXT,
                    status TEXT NOT NULL CHECK(status IN ('present', 'absent')),
                    PRIMARY KEY (session_id, student_id)
                );
                CREATE TABLE IF NOT EXISTS student_accounts (
                    student_id INTEGER PRIMARY KEY REFERENCES students(id) ON DELETE CASCADE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS instructor_accounts (
                    id INTEGER PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection

    def register_student(self, name: str, institution: str, enrollment_number: str,
                         photo: bytes) -> int:
        name, institution = name.strip(), institution.strip()
        if not name or not institution:
            raise ValueError("Name and school/college are required.")
        if len(name) > 200 or len(institution) > 200:
            raise ValueError("Name and school/college must be at most 200 characters.")
        enrollment_number = _validate_enrollment_number(enrollment_number)
        if not photo:
            raise ValueError("Provide a student photo before registering.")
        face = _prepare_face(photo, self.cfg)
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO students
                   (name, institution, enrollment_number, photo, face_template, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, institution, enrollment_number, photo, face.tobytes(),
                 datetime.now().isoformat(timespec="seconds")),
            )
            return int(cursor.lastrowid)

    def list_students(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, name, institution, enrollment_number, created_at
                   FROM students ORDER BY name COLLATE NOCASE"""
            ).fetchall()
        return [dict(row) for row in rows]

    def create_student_account(self, enrollment_number: str, password: str) -> int:
        enrollment_number = _validate_enrollment_number(enrollment_number)
        if len(password) < _MIN_PASSWORD_LENGTH:
            raise ValueError(f"Use a password with at least {_MIN_PASSWORD_LENGTH} characters.")
        if len(password) > _MAX_PASSWORD_LENGTH:
            raise ValueError(f"Use a password with at most {_MAX_PASSWORD_LENGTH} characters.")
        if not password.strip():
            raise ValueError("Password cannot contain only whitespace.")
        with self._connection() as connection:
            student = connection.execute(
                "SELECT id FROM students WHERE enrollment_number = ? COLLATE NOCASE",
                (enrollment_number,),
            ).fetchone()
            if student is None:
                raise ValueError(
                    "We couldn't create an account with those details. Check your enrollment "
                    "number or contact your instructor."
                )
            student_id = int(student["id"])
            if connection.execute(
                "SELECT 1 FROM student_accounts WHERE student_id = ?", (student_id,)
            ).fetchone():
                raise ValueError(
                    "We couldn't create an account with those details. Check your enrollment "
                    "number or contact your instructor."
                )
            salt = secrets.token_bytes(_PASSWORD_SALT_BYTES)
            connection.execute(
                """INSERT INTO student_accounts (student_id, password_salt, password_hash, created_at)
                   VALUES (?, ?, ?, ?)""",
                (student_id, salt, _hash_password(password, salt),
                 datetime.now().isoformat(timespec="seconds")),
            )
        return student_id

    def authenticate_student(self, enrollment_number: str, password: str) -> int | None:
        if not enrollment_number.isprintable():
            return None
        enrollment_number = enrollment_number.strip()
        if not enrollment_number or len(enrollment_number) > _MAX_ENROLLMENT_LENGTH:
            return None
        if len(password) > _MAX_PASSWORD_LENGTH:
            return None
        with self._connection() as connection:
            row = connection.execute(
                """SELECT a.student_id, a.password_salt, a.password_hash
                   FROM student_accounts a
                   JOIN students s ON s.id = a.student_id
                   WHERE s.enrollment_number = ? COLLATE NOCASE""",
                (enrollment_number,),
            ).fetchone()
        if row is None:
            salt = bytes(_PASSWORD_SALT_BYTES)
            expected = _hash_password(password, salt)
            hmac.compare_digest(expected, bytes(_PASSWORD_HASH_BYTES))
            return None
        actual = _hash_password(password, bytes(row["password_salt"]))
        if not hmac.compare_digest(actual, bytes(row["password_hash"])):
            return None
        return int(row["student_id"])

    def get_student_profile(self, student_id: int) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT id, name, institution, enrollment_number
                   FROM students WHERE id = ?""",
                (student_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def student_attendance(self, student_id: int, limit: int = 100) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT r.presence_seconds, r.first_seen, r.last_seen, r.status,
                          a.started_at AS session_started_at, a.ended_at AS session_ended_at
                   FROM attendance_records r
                   JOIN attendance_sessions a ON a.id = r.session_id
                   WHERE r.student_id = ?
                   ORDER BY a.id DESC LIMIT ?""",
                (student_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def has_instructor_accounts(self) -> bool:
        with self._connection() as connection:
            return connection.execute(
                "SELECT EXISTS(SELECT 1 FROM instructor_accounts)"
            ).fetchone()[0] == 1

    @staticmethod
    def _validate_instructor_account(full_name: str, username: str, password: str):
        full_name, username = full_name.strip(), username.strip()
        if not full_name or len(full_name) < 2 or len(full_name) > 100:
            raise ValueError("Enter a name between 2 and 100 characters.")
        if not _INSTRUCTOR_USERNAME_PATTERN.fullmatch(username):
            raise ValueError(
                "Username must be 3–32 characters and use only letters, numbers, '.', '_' or '-'."
            )
        if len(password) < _MIN_PASSWORD_LENGTH:
            raise ValueError(f"Use a password with at least {_MIN_PASSWORD_LENGTH} characters.")
        if len(password) > _MAX_PASSWORD_LENGTH:
            raise ValueError(f"Use a password with at most {_MAX_PASSWORD_LENGTH} characters.")
        if not password.strip():
            raise ValueError("Password cannot contain only whitespace.")
        return full_name, username

    def _insert_instructor_account(self, connection, full_name: str, username: str,
                                   password: str) -> int:
        salt = secrets.token_bytes(_PASSWORD_SALT_BYTES)
        cursor = connection.execute(
            """INSERT INTO instructor_accounts
               (full_name, username, password_salt, password_hash, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (full_name, username, salt, _hash_password(password, salt),
             datetime.now().isoformat(timespec="seconds")),
        )
        return int(cursor.lastrowid)

    def create_first_instructor_account(self, full_name: str, username: str,
                                        password: str) -> int:
        full_name, username = self._validate_instructor_account(full_name, username, password)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT EXISTS(SELECT 1 FROM instructor_accounts)"
            ).fetchone()[0]:
                raise ValueError("Initial instructor setup has already been completed.")
            return self._insert_instructor_account(connection, full_name, username, password)

    def create_instructor_account(self, full_name: str, username: str, password: str) -> int:
        full_name, username = self._validate_instructor_account(full_name, username, password)
        with self._connection() as connection:
            return self._insert_instructor_account(connection, full_name, username, password)

    def authenticate_instructor(self, username: str, password: str) -> dict | None:
        username = username.strip()
        if not username or len(username) > 32 or len(password) > _MAX_PASSWORD_LENGTH:
            return None
        with self._connection() as connection:
            row = connection.execute(
                """SELECT id, full_name, username, password_salt, password_hash
                   FROM instructor_accounts WHERE username = ? COLLATE NOCASE""",
                (username,),
            ).fetchone()
        if row is None:
            salt = bytes(_PASSWORD_SALT_BYTES)
            hmac.compare_digest(_hash_password(password, salt), bytes(_PASSWORD_HASH_BYTES))
            return None
        actual = _hash_password(password, bytes(row["password_salt"]))
        if not hmac.compare_digest(actual, bytes(row["password_hash"])):
            return None
        return {
            "id": int(row["id"]),
            "full_name": row["full_name"],
            "username": row["username"],
        }

    def list_instructor_accounts(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, full_name, username, created_at
                   FROM instructor_accounts ORDER BY full_name COLLATE NOCASE"""
            ).fetchall()
        return [dict(row) for row in rows]

    def train_recognizer(self):
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError(
                "Face recognition requires opencv-contrib-python. "
                "Install the project dependencies with: pip install -r requirements.txt"
            )
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, face_template FROM students ORDER BY id"
            ).fetchall()
        if not rows:
            raise ValueError("Register at least one student before starting attendance.")
        faces = [
            np.frombuffer(row["face_template"], dtype=np.uint8).reshape(_FACE_SIZE[::-1])
            for row in rows
        ]
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.asarray([int(row["id"]) for row in rows], dtype=np.int32))
        return recognizer

    def recognize(self, frame: np.ndarray, recognizer, confidence_threshold: float):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._frontal.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.cfg.min_face_size, self.cfg.min_face_size),
        )
        matches = []
        for x, y, width, height in faces:
            face = cv2.resize(
                cv2.equalizeHist(gray[y:y + height, x:x + width]), _FACE_SIZE,
                interpolation=cv2.INTER_AREA,
            )
            label, confidence = recognizer.predict(face)
            if confidence <= confidence_threshold:
                matches.append((int(label), (int(x), int(y), int(width), int(height))))
        return matches

    def save_attendance_session(
        self,
        started_at: datetime,
        ended_at: datetime,
        required_presence_seconds: float,
        presence: dict[int, float],
        first_seen: dict[int, float],
        last_seen: dict[int, float],
    ) -> list[dict]:
        if required_presence_seconds <= 0:
            raise ValueError("Required attendance duration must be greater than zero.")
        if ended_at < started_at:
            raise ValueError("Attendance session end time cannot be before its start time.")
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO attendance_sessions
                   (started_at, ended_at, required_presence_seconds) VALUES (?, ?, ?)""",
                (started_at.isoformat(timespec="seconds"), ended_at.isoformat(timespec="seconds"),
                 required_presence_seconds),
            )
            session_id = int(cursor.lastrowid)
            students = connection.execute("SELECT id FROM students ORDER BY id").fetchall()
            records = []
            for row in students:
                student_id = int(row["id"])
                seconds = round(presence.get(student_id, 0.0), 1)
                first = first_seen.get(student_id)
                last = last_seen.get(student_id)
                record = {
                    "session_id": session_id,
                    "student_id": student_id,
                    "presence_seconds": seconds,
                    "first_seen": (started_at + timedelta(seconds=first)).isoformat(timespec="seconds")
                    if first is not None else None,
                    "last_seen": (started_at + timedelta(seconds=last)).isoformat(timespec="seconds")
                    if last is not None else None,
                    "status": "present" if seconds >= required_presence_seconds else "absent",
                }
                records.append(record)
                connection.execute(
                    """INSERT INTO attendance_records
                       (session_id, student_id, presence_seconds, first_seen, last_seen, status)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session_id, student_id, seconds, record["first_seen"], record["last_seen"],
                     record["status"]),
                )
        return records

    def recent_attendance(self, limit: int = 100) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT s.name, s.institution, s.enrollment_number, r.presence_seconds,
                          r.first_seen, r.last_seen, r.status, a.started_at AS session_started_at
                   FROM attendance_records r
                   JOIN students s ON s.id = r.student_id
                   JOIN attendance_sessions a ON a.id = r.session_id
                   ORDER BY a.id DESC, s.name COLLATE NOCASE LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


def run_camera_attendance(
    registry: StudentRegistry,
    camera_index: int = 0,
    duration_seconds: float = 2700,
    required_presence_seconds: float = 30,
    confidence_threshold: float = _DEFAULT_CONFIDENCE,
    progress_callback: Callable[[float, float], None] | None = None,
) -> list[dict]:
    """Run timed local-camera recognition and persist one attendance record per student."""
    if duration_seconds <= 0:
        raise ValueError("Session duration must be greater than zero.")
    if required_presence_seconds <= 0:
        raise ValueError("Required attendance duration must be greater than zero.")
    if not 0 <= confidence_threshold <= 100:
        raise ValueError("Recognition threshold must be between 0 and 100.")
    recognizer = registry.train_recognizer()
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {camera_index}. Check the camera index and permissions.")

    started_at = datetime.now()
    start = time.monotonic()
    next_sample = start
    previous_sample: float | None = None
    period = 1.0 / registry.cfg.process_fps
    presence: dict[int, float] = {}
    first_seen: dict[int, float] = {}
    last_seen: dict[int, float] = {}
    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= duration_seconds:
                break
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames; attendance was not saved.")
            now = time.monotonic()
            if now < next_sample:
                continue
            interval = period if previous_sample is None else min(now - previous_sample, period * 2)
            previous_sample = now
            next_sample = now + period
            elapsed = now - start
            seen_this_frame = {
                student_id
                for student_id, _ in registry.recognize(frame, recognizer, confidence_threshold)
            }
            for student_id in seen_this_frame:
                presence[student_id] = presence.get(student_id, 0.0) + interval
                first_seen.setdefault(student_id, elapsed)
                last_seen[student_id] = elapsed
            if progress_callback is not None:
                progress_callback(min(elapsed / duration_seconds, 1.0), elapsed)
    finally:
        camera.release()

    ended_at = datetime.now()
    return registry.save_attendance_session(
        started_at, ended_at, required_presence_seconds, presence, first_seen, last_seen
    )
