import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pytest

from attendance_agent import registry as registry_module
from attendance_agent.registry import StudentRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        registry_module, "_prepare_face", lambda photo, cfg: np.full((160, 160), 127, np.uint8)
    )
    return StudentRegistry(tmp_path / "attendance.sqlite3")


def add_student(registry, enrollment_number, name="A Student"):
    return registry.register_student(name, "Example College", enrollment_number, b"photo")


def test_registration_persists_student_details_and_unique_enrollment(registry):
    student_id = add_student(registry, "ENR-001", "  A Student  ")

    students = registry.list_students()
    assert len(students) == 1
    assert {key: students[0][key] for key in ("id", "name", "institution", "enrollment_number")} == {
        "id": student_id,
        "name": "A Student",
        "institution": "Example College",
        "enrollment_number": "ENR-001",
    }
    datetime.fromisoformat(students[0]["created_at"])
    with pytest.raises(sqlite3.IntegrityError):
        add_student(registry, "enr-001", "Another Student")


def test_face_recognizer_trains_from_registered_templates(registry):
    add_student(registry, "ENR-001")

    assert not registry.train_recognizer().empty()


def test_attendance_session_saves_present_and_absent_students(registry):
    present_id = add_student(registry, "ENR-001", "Present Student")
    absent_id = add_student(registry, "ENR-002", "Absent Student")
    started = datetime(2026, 10, 19, 10, 0)

    records = registry.save_attendance_session(
        started,
        started + timedelta(minutes=1),
        required_presence_seconds=30,
        presence={present_id: 32.6},
        first_seen={present_id: 1.2},
        last_seen={present_id: 59.8},
    )

    assert [record["status"] for record in records] == ["present", "absent"]
    assert records[0]["presence_seconds"] == 32.6
    history = registry.recent_attendance()
    assert [record["status"] for record in history] == ["absent", "present"]
    assert history[1]["enrollment_number"] == "ENR-001"
    assert history[1]["first_seen"] == "2026-10-19T10:00:01"


def test_registration_rejects_incomplete_details(registry):
    with pytest.raises(ValueError, match="required"):
        registry.register_student("", "Example College", "ENR-001", b"photo")


def test_student_signup_requires_registered_enrollment_and_unique_account(registry):
    student_id = add_student(registry, "ENR-001")

    with pytest.raises(ValueError, match="enrollment"):
        registry.create_student_account("UNKNOWN", "a-strong-password")

    assert registry.create_student_account("enr-001", "a-strong-password") == student_id
    assert registry.authenticate_student("ENR-001", "a-strong-password") == student_id
    assert registry.authenticate_student("ENR-001", "wrong-password") is None
    assert registry.authenticate_student("UNKNOWN", "a-strong-password") is None
    with registry._connection() as connection:
        account = connection.execute(
            "SELECT password_salt, password_hash FROM student_accounts WHERE student_id = ?",
            (student_id,),
        ).fetchone()
    assert len(account["password_salt"]) == 16
    assert len(account["password_hash"]) == 64
    assert account["password_hash"] != b"a-strong-password"
    with pytest.raises(ValueError, match="account"):
        registry.create_student_account("ENR-001", "another-strong-password")


def test_student_account_rejects_invalid_enrollment_numbers(registry):
    with pytest.raises(ValueError, match="required"):
        registry.create_student_account("   ", "a-strong-password")
    with pytest.raises(ValueError, match="128 characters"):
        registry.create_student_account("E" * 129, "a-strong-password")
    with pytest.raises(ValueError, match="unsupported characters"):
        registry.create_student_account("ENR-001\n", "a-strong-password")
    assert registry.authenticate_student("   ", "a-strong-password") is None


def test_student_password_minimum_and_attendance_history_isolation(registry):
    first_id = add_student(registry, "ENR-001", "First Student")
    second_id = add_student(registry, "ENR-002", "Second Student")
    with pytest.raises(ValueError, match="12 characters"):
        registry.create_student_account("ENR-001", "short")
    with pytest.raises(ValueError, match="only whitespace"):
        registry.create_student_account("ENR-001", " " * 12)
    with pytest.raises(ValueError, match="1024 characters"):
        registry.create_student_account("ENR-001", "x" * 1025)
    assert registry.authenticate_student("ENR-001", "x" * 1025) is None

    started = datetime(2026, 10, 19, 10, 0)
    registry.save_attendance_session(
        started,
        started + timedelta(minutes=1),
        30,
        presence={first_id: 35, second_id: 5},
        first_seen={first_id: 1, second_id: 2},
        last_seen={first_id: 50, second_id: 20},
    )

    assert registry.get_student_profile(first_id)["name"] == "First Student"
    first_history = registry.student_attendance(first_id)
    second_history = registry.student_attendance(second_id)
    assert len(first_history) == len(second_history) == 1
    assert first_history[0]["status"] == "present"
    assert second_history[0]["status"] == "absent"
    assert "enrollment_number" not in first_history[0]


def test_instructor_first_account_bootstrap_and_authentication(registry):
    assert not registry.has_instructor_accounts()

    instructor_id = registry.create_first_instructor_account(
        "  Ada Instructor  ", "  Ada.Admin  ", "a-strong-instructor-password"
    )

    assert registry.has_instructor_accounts()
    assert registry.authenticate_instructor("ada.admin", "a-strong-instructor-password") == {
        "id": instructor_id,
        "full_name": "Ada Instructor",
        "username": "Ada.Admin",
    }
    assert registry.authenticate_instructor("ada.admin", "incorrect-password") is None
    assert registry.authenticate_instructor("unknown", "a-strong-instructor-password") is None
    assert registry.list_instructor_accounts()[0]["id"] == instructor_id
    with pytest.raises(ValueError, match="already been completed"):
        registry.create_first_instructor_account(
            "Another Instructor", "another.user", "another-strong-password"
        )

    with registry._connection() as connection:
        account = connection.execute(
            "SELECT password_salt, password_hash FROM instructor_accounts WHERE id = ?",
            (instructor_id,),
        ).fetchone()
    assert len(account["password_salt"]) == 16
    assert len(account["password_hash"]) == 64
    assert account["password_hash"] != b"a-strong-instructor-password"


def test_instructor_account_validation_and_username_uniqueness(registry):
    registry.create_first_instructor_account(
        "First Instructor", "first.user", "a-strong-instructor-password"
    )
    with pytest.raises(sqlite3.IntegrityError):
        registry.create_instructor_account(
            "Another Instructor", "FIRST.USER", "another-strong-password"
        )
    with pytest.raises(ValueError, match="name between"):
        registry.create_instructor_account("", "second.user", "another-strong-password")
    with pytest.raises(ValueError, match="Username"):
        registry.create_instructor_account(
            "Another Instructor", "bad username", "another-strong-password"
        )
    with pytest.raises(ValueError, match="12 characters"):
        registry.create_instructor_account("Another Instructor", "second.user", "short")
    with pytest.raises(ValueError, match="only whitespace"):
        registry.create_instructor_account("Another Instructor", "second.user", " " * 12)
    with pytest.raises(ValueError, match="1024 characters"):
        registry.create_instructor_account("Another Instructor", "second.user", "x" * 1025)
    assert registry.authenticate_instructor(" ", "password") is None
