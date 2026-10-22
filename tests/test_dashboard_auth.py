from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from attendance_agent import registry as registry_module
from attendance_agent.registry import StudentRegistry

_APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _prepare_app(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ATTENDANCE_DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        registry_module, "_prepare_face", lambda photo, cfg: np.full((160, 160), 127, np.uint8)
    )
    return StudentRegistry(data_dir / "student_registry.sqlite3")


def test_dashboard_opens_without_student_or_instructor_authentication(tmp_path, monkeypatch):
    _prepare_app(tmp_path, monkeypatch)

    app = AppTest.from_file(_APP_PATH).run()

    assert not app.exception
    assert not app.tabs
    assert not app.session_state.get("account_type")
    assert not any(
        "password" in item.label.casefold()
        or "username" in item.label.casefold()
        or "enrollment number" in item.label.casefold()
        for item in app.text_input
    )
    assert "Welcome to your classroom workspace" in [item.value for item in app.subheader]
    assert any(button.key == "open_feature_registration" for button in app.button)
    assert not any(button.label == "Sign out" for button in app.button)


def test_dashboard_features_are_open_without_login(tmp_path, monkeypatch):
    _prepare_app(tmp_path, monkeypatch)
    app = AppTest.from_file(_APP_PATH).run()

    pages = [
        ("attendance", "Session attendance"),
        ("registration", "Student directory"),
        ("live_attendance", "Start live attendance"),
        ("engagement", "Session engagement"),
        ("instructor_notes", "Instructor notes"),
        ("learner_view", "Individual session view"),
    ]
    for page, heading in pages:
        app.query_params["feature"] = page
        app.run()

        assert not app.exception, [str(item.value) for item in app.exception]
        assert heading in [item.value for item in app.subheader]
        assert not app.tabs


def test_existing_student_accounts_do_not_gate_dashboard(tmp_path, monkeypatch):
    registry = _prepare_app(tmp_path, monkeypatch)
    student_id = registry.register_student("Student One", "Example College", "ENR-001", b"photo")
    registry.create_student_account("ENR-001", "a-strong-test-password")

    app = AppTest.from_file(_APP_PATH).run()

    assert not app.exception
    assert app.session_state.get("student_id") != student_id
    assert "Welcome to your classroom workspace" in [item.value for item in app.subheader]
