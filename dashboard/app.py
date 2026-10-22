"""Optional Streamlit dashboard (Activity 6.2).

Run:  streamlit run dashboard/app.py
Reads saved sessions from ./output (or generates a synthetic demo session).
"""
import html
import json
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attendance_agent.config import Config  # noqa: E402
from attendance_agent.demo import build_demo  # noqa: E402
from attendance_agent.pipeline import SessionPipeline  # noqa: E402
from attendance_agent.registry import StudentRegistry, run_camera_attendance  # noqa: E402

st.set_page_config(page_title="Attendance & Engagement Agent", layout="wide")
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: rgba(245, 247, 251, 0.92); }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color: #f8fafc; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aebbd0; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #e5e7eb; }
    [data-testid="stSidebar"] input { color: #111827 !important; background: #ffffff; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #111827 !important; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    .app-brand { color: #93c5fd; font-size: .78rem; font-weight: 700; letter-spacing: .12em;
                 text-transform: uppercase; margin: 0 0 .35rem; }
    .hero { background: linear-gradient(120deg, #172554 0%, #1e3a8a 58%, #2563eb 100%);
            border-radius: 18px; padding: 1.65rem 2rem; margin: 0 0 1.4rem; color: white; }
    .hero h1 { color: white; font-size: 2rem; line-height: 1.2; margin: 0 0 .45rem; }
    .hero p { color: #dbeafe; font-size: 1rem; margin: 0; }
    .feature-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
                    gap: 1rem; margin: .65rem 0 1.5rem; }
    .feature-card { min-height: 148px; background: #fff; border: 1px solid #e5eaf2;
                    border-radius: 15px; padding: 1.2rem 1.25rem;
                    box-shadow: 0 3px 12px rgba(15, 23, 42, .045);
                    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease; }
    .feature-card:hover { transform: translateY(-2px); border-color: #bfdbfe;
                          box-shadow: 0 8px 20px rgba(37, 99, 235, .09); }
    .st-key-feature_cards [data-testid="stButton"] > button {
        min-height: 148px; height: 100%; padding: 1.2rem 1.25rem; text-align: left;
        white-space: pre-wrap; align-items: flex-start; border: 1px solid #e5eaf2;
        border-radius: 15px; background: #fff; color: #172554;
        box-shadow: 0 3px 12px rgba(15, 23, 42, .045);
    }
    .st-key-feature_cards [data-testid="stButton"] > button:hover {
        border-color: #bfdbfe; color: #172554;
        box-shadow: 0 8px 20px rgba(37, 99, 235, .09);
    }
    .st-key-feature_cards [data-testid="stButton"] > button p {
        white-space: pre-wrap; text-align: left;
    }
    .st-key-feature_cards [class*="st-key-feature_card_"] { position: relative; }
    .st-key-feature_cards [class*="st-key-feature_card_"]:hover,
    .st-key-feature_cards [class*="st-key-feature_card_"]:focus-within { z-index: 5; }
    .feature-hover-preview {
        display: none; position: absolute; inset: 0; z-index: 2; pointer-events: none;
        padding: 1.2rem 1.25rem; border: 1px solid #bfdbfe; border-radius: 15px;
        background: #fff; box-shadow: 0 8px 20px rgba(37, 99, 235, .12);
        color: #334155; line-height: 1.55;
    }
    .feature-hover-preview strong { display: block; color: #172554; margin-bottom: .55rem; }
    .st-key-feature_cards [class*="st-key-feature_card_"]:hover .feature-hover-preview,
    .st-key-feature_cards [class*="st-key-feature_card_"]:focus-within .feature-hover-preview {
        display: block;
    }
    .feature-icon { display: inline-flex; align-items: center; justify-content: center;
                    width: 2.35rem; height: 2.35rem; border-radius: 11px;
                    background: #eff6ff; font-size: 1.2rem; margin-bottom: .75rem; }
    .feature-card h3 { color: #172554; font-size: 1rem; margin: 0 0 .38rem; }
    .feature-card p { color: #64748b; font-size: .9rem; line-height: 1.5; margin: 0; }
    .home-section-caption { color: #64748b; margin-top: -.65rem; margin-bottom: .85rem; }
    @media (max-width: 850px) { .feature-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 560px) { .feature-grid { grid-template-columns: 1fr; }
                                 .feature-card { min-height: 0; } }
    [data-testid="stMetric"] { background: white; border: 1px solid #e5eaf2; border-radius: 14px;
                               padding: 1rem 1.1rem; box-shadow: 0 2px 8px rgba(15, 23, 42, .035); }
    [data-testid="stMetricLabel"] { color: #64748b; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #172554; }
    [data-testid="stTabs"] button { font-weight: 600; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #1d4ed8; }
    [data-testid="stDataFrame"] { border: 1px solid #e5eaf2; border-radius: 12px; overflow: hidden; }
    div.stButton > button[kind="primary"] { border-radius: 9px; font-weight: 650; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    h2, h3 { color: #172554; }
</style>
""", unsafe_allow_html=True)

FEATURE_PAGES = [
    ("attendance", "📋", "Presence-based attendance",
     "Track cumulative face presence, thresholds, and attendance exports."),
    ("registration", "🪪", "Student registry",
     "Manage consent-based student profiles, enrollment numbers, and face photos."),
    ("live_attendance", "📷", "Live camera sessions",
     "Run timed webcam attendance and review recognized presence."),
    ("engagement", "📈", "Engagement trends",
     "Explore attention, movement, room noise, and speech signals over time."),
    ("instructor_notes", "💡", "Instructor notes",
     "Review evidence-grounded prompts about session trends."),
    ("learner_view", "🔎", "Individual learner summaries",
     "Review individual reflections without grades or peer comparisons."),
]

FEATURE_PAGE_TITLES = {page: title for page, _, title, _ in FEATURE_PAGES}
FEATURE_PREVIEWS = {
    "attendance": "Review present and absent students, face-presence duration, and downloadable session attendance records.",
    "registration": "Register students with their institution, unique enrollment number, and consented face photo.",
    "live_attendance": "Choose the camera and attendance threshold, then run and save a timed face-recognition session.",
    "engagement": "Explore class engagement, room-noise, speech-activity, and participant trends across the session.",
    "instructor_notes": "Read evidence-grounded session prompts with the observations that informed each suggestion.",
    "learner_view": "Review a selected learner's attendance and session reflection without rankings.",
}


def render_feature_cards() -> None:
    with st.container(key="feature_cards"):
        for row_start in range(0, len(FEATURE_PAGES), 3):
            columns = st.columns(3)
            for column, (page, icon, title, description) in zip(
                columns, FEATURE_PAGES[row_start:row_start + 3]
            ):
                with column:
                    with st.container(key=f"feature_card_{page}"):
                        if st.button(
                            f"{icon}  **{title}**\n\n{description}\n\nOpen feature →",
                            key=f"open_feature_{page}",
                            use_container_width=True,
                        ):
                            st.session_state["instructor_page"] = page
                            st.rerun()
                        st.markdown(
                            '<div class="feature-hover-preview">'
                            f"<strong>{html.escape(title)}</strong>"
                            f"{html.escape(FEATURE_PREVIEWS[page])}</div>",
                            unsafe_allow_html=True,
                        )


OUT = Path(os.environ.get("ATTENDANCE_DATA_DIR", "output"))
st.sidebar.markdown("### Dashboard")
st.sidebar.caption("Session insights and student records stay on this machine.")
st.sidebar.warning("No sign-in is required. Anyone with dashboard access can view and change records.")
registry = StudentRegistry(OUT / "student_registry.sqlite3")
st.session_state["instructor_page"] = st.session_state.get("instructor_page", "home")
requested_feature = st.query_params.get("feature")
if requested_feature is not None:
    if requested_feature == "home":
        st.session_state["instructor_page"] = "home"
        del st.query_params["feature"]
    elif requested_feature in FEATURE_PAGE_TITLES:
        st.session_state["instructor_page"] = requested_feature
    else:
        st.session_state["instructor_page"] = "home"
        del st.query_params["feature"]
sessions = sorted([p for p in OUT.glob("session_*") if p.is_dir()], reverse=True) if OUT.exists() else []
choice = st.sidebar.selectbox("Choose a session", ["(synthetic demo)"] + [p.name for p in sessions])
registered = registry.list_students()
st.sidebar.caption(f"{len(sessions)} saved session{'s' if len(sessions) != 1 else ''} · {len(registered)} registered students")


@st.cache_data
def load_demo():
    cfg = Config()
    pipe = SessionPipeline(cfg)
    obs, dt, dur, audio = build_demo(cfg)
    r = pipe.analyse(obs, dt, dur, audio)
    return (pd.DataFrame(r.attendance), r.class_timeline, r.participant_scores.drop(columns=["components"]),
            r.suggestions, r.student_analytics, r.summary)


def load_saved(folder: Path):
    rd = lambda n: pd.read_csv(folder / n) if (folder / n).exists() else pd.DataFrame()
    js = lambda n: json.loads((folder / n).read_text())
    return (rd("attendance.csv"), rd("class_timeline.csv"), rd("participant_engagement.csv"),
            js("suggestions.json"), js("student_analytics.json"), js("summary.json"))


att, timeline, scores, suggestions, students, summary = (
    load_demo() if choice == "(synthetic demo)" else load_saved(OUT / choice))

current_page = st.session_state.get("instructor_page", "home")
if current_page != "home" and current_page not in FEATURE_PAGE_TITLES:
    current_page = "home"
    st.session_state["instructor_page"] = "home"

if current_page == "home":
    st.markdown("""
    <div class="hero">
      <div class="app-brand">Classroom dashboard · local processing</div>
      <h1>Attendance &amp; Engagement</h1>
      <p>Review class sessions, explore engagement patterns, and manage student attendance.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if st.button("← Home", key="feature_page_home"):
        st.session_state["instructor_page"] = "home"
        st.query_params.clear()
        st.rerun()
    st.markdown(f"""
    <div class="hero">
      <div class="app-brand">Classroom dashboard · feature page</div>
      <h1>{html.escape(FEATURE_PAGE_TITLES[current_page])}</h1>
      <p>Selected session: {html.escape("Synthetic demo" if choice == "(synthetic demo)" else choice)}</p>
    </div>
    """, unsafe_allow_html=True)

with_home = current_page == "home"
if with_home:
    st.subheader("Welcome to your classroom workspace")
    session_label = "Synthetic demo" if choice == "(synthetic demo)" else choice
    st.markdown(
        f'<p class="home-section-caption">Selected session: <strong>{html.escape(session_label)}</strong>'
        " · Select a feature to open its dedicated page.</p>",
        unsafe_allow_html=True,
    )
    render_feature_cards()

    st.subheader("Session overview")
    st.caption("A quick summary of the selected class session. Engagement is a trend, not a grade.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Session length", f"{summary['duration_seconds'] / 60:.0f} min")
    metric_columns[1].metric(
        "Attendance",
        f"{summary['participants_present']} / {summary['participants_detected']}",
        help="Participants above the configured cumulative presence threshold.",
    )
    metric_columns[2].metric(
        "Mean engagement",
        f"{summary['mean_class_engagement']:.1f}" if summary["mean_class_engagement"] is not None else "—",
        help="Engagement is a session trend, not a grade or assessment.",
    )
    noise = summary.get("mean_noise_db")
    metric_columns[3].metric("Mean room noise", f"{noise:.1f} dBFS" if noise is not None else "—")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Attendance at a glance")
        st.caption("Presence is cumulative; a brief face detection alone does not count as attendance.")
        if att.empty:
            st.info("No attendance records are available for this session.")
        else:
            st.dataframe(
                att[["participant_id", "status", "presence_seconds", "presence_percent",
                     "first_seen", "last_seen"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "participant_id": st.column_config.TextColumn("Participant"),
                    "status": st.column_config.TextColumn("Status"),
                    "presence_seconds": st.column_config.NumberColumn("Time present", format="%.1f s"),
                    "presence_percent": st.column_config.ProgressColumn(
                        "Session coverage", min_value=0, max_value=100, format="%.0f%%"
                    ),
                    "first_seen": st.column_config.TextColumn("First seen"),
                    "last_seen": st.column_config.TextColumn("Last seen"),
                },
            )
    with right:
        st.subheader("Engagement over time")
        if not timeline.empty and "class_engagement" in timeline:
            chart = timeline.copy()
            chart["minute"] = chart["t_start"] / 60
            st.area_chart(chart.set_index("minute")["class_engagement"], height=245)
            st.caption("Per-window class trend · not an individual score.")
        else:
            st.info("No engagement timeline is available.")

    if suggestions:
        st.subheader("Instructor notes")
        for suggestion in suggestions[:2]:
            st.info(f"**{suggestion['priority'].upper()} · {suggestion['time_range']}**  \n"
                    f"{suggestion['suggestion']}")
    else:
        st.success("No instructor notes were generated for this session.")

if current_page == "attendance":
    st.subheader("Session attendance")
    st.caption("Attendance is based on cumulative presence. Review records before using them officially.")
    if att.empty:
        st.info("No attendance records are available for this session.")
    else:
        filter_column, summary_column, download_column = st.columns([1, 1, 1])
        status_filter = filter_column.selectbox(
            "Show students", ["Everyone", "Present", "Absent"], key="attendance_status_filter"
        )
        present_count = int((att["status"] == "present").sum())
        summary_column.metric("Marked present", f"{present_count} / {len(att)}")
        download_column.download_button(
            "Download attendance CSV",
            data=att.to_csv(index=False).encode("utf-8"),
            file_name=f"{choice.removeprefix('session_')}_attendance.csv",
            mime="text/csv",
            use_container_width=True,
        )
        attendance_view = att
        if status_filter != "Everyone":
            attendance_view = att[att["status"].str.casefold() == status_filter.casefold()]
        st.dataframe(
            attendance_view,
            hide_index=True,
            use_container_width=True,
            column_config={
                "participant_id": st.column_config.TextColumn("Participant"),
                "status": st.column_config.TextColumn("Status"),
                "presence_seconds": st.column_config.NumberColumn("Time present", format="%.1f s"),
                "presence_percent": st.column_config.ProgressColumn(
                    "Session coverage", min_value=0, max_value=100, format="%.0f%%"
                ),
                "first_seen": st.column_config.TextColumn("First seen"),
                "last_seen": st.column_config.TextColumn("Last seen"),
            },
        )

if current_page == "registration":
    st.subheader("Student directory")
    st.caption(
        "Add an authorized student record. The photo and face template are stored in the local database."
    )
    details, photo_column = st.columns([1, 1.15], gap="large")
    with details:
        st.markdown("#### Student details")
        name = st.text_input("Full name", key="student_name", placeholder="e.g. Alex Morgan")
        institution = st.text_input(
            "School or college", key="student_institution", placeholder="Institution name"
        )
        enrollment_number = st.text_input(
            "Enrollment number", key="student_enrollment", placeholder="Unique student ID"
        )
        consent = st.checkbox("I have the student's permission to store and use this face photo.")
    with photo_column:
        st.markdown("#### Student photo")
        image_source = st.radio(
            "Choose a photo source", ["Take a photo", "Upload a photo"], horizontal=True
        )
        if image_source == "Take a photo":
            photo_file = st.camera_input("Capture a clear, front-facing photo", key="student_camera")
        else:
            photo_file = st.file_uploader(
                "Upload a clear, front-facing photo", type=["jpg", "jpeg", "png"],
                key="student_photo",
            )
        if photo_file is not None:
            st.image(photo_file, caption="Photo to be registered", width=220)
    if st.button("Register student", type="primary"):
        if not consent:
            st.error("Confirm the student's permission before registering.")
        elif photo_file is None:
            st.error("Capture or upload a photo before registering.")
        else:
            try:
                student_id = registry.register_student(
                    name, institution, enrollment_number, photo_file.getvalue()
                )
            except ValueError as error:
                st.error(str(error))
            except sqlite3.IntegrityError:
                st.error("That enrollment number is already registered.")
            else:
                st.success(f"Student registered (database ID {student_id}).")

    st.divider()
    st.subheader(f"Registered students · {len(registered)}")
    if registered:
        st.dataframe(
            pd.DataFrame(registered),
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": None,
                "name": st.column_config.TextColumn("Student"),
                "institution": st.column_config.TextColumn("School / college"),
                "enrollment_number": st.column_config.TextColumn("Enrollment no."),
                "created_at": st.column_config.DatetimeColumn("Registered"),
            },
        )
    else:
        st.info("Your directory is empty. Add a student above to prepare for face attendance.")

if current_page == "live_attendance":
    st.subheader("Start live attendance")
    st.caption("Recognized presence accumulates during the class and is saved when the session ends.")
    st.warning(
        "Face recognition can be wrong or miss students. Review saved records before treating them "
        "as official attendance."
    )
    st.caption(
        "Local processing only: the webcam feed is not saved or sent to a service. "
        "Only recognized presence duration and attendance timestamps are stored."
    )
    camera_col, duration_col = st.columns(2)
    camera_index = camera_col.number_input("Camera index", min_value=0, step=1, value=0)
    duration_minutes = duration_col.number_input(
        "Session duration (minutes)", min_value=1, max_value=240, value=45, step=1
    )
    presence_col, threshold_col = st.columns(2)
    required_seconds = presence_col.number_input(
        "Seconds required to mark present", min_value=1, max_value=3600, value=30, step=5
    )
    confidence_threshold = threshold_col.slider(
        "Face-match confidence limit (lower is stricter)", min_value=20, max_value=100, value=60
    )
    if st.button("Start camera attendance", type="primary"):
        if not registered:
            st.error("Register at least one student before starting attendance.")
        else:
            progress = st.progress(0.0, text="Preparing local face recognizer…")
            try:
                records = run_camera_attendance(
                    registry,
                    camera_index=int(camera_index),
                    duration_seconds=float(duration_minutes * 60),
                    required_presence_seconds=float(required_seconds),
                    confidence_threshold=float(confidence_threshold),
                    progress_callback=lambda fraction, elapsed: progress.progress(
                        fraction, text=f"Attendance running: {elapsed:.0f}s elapsed"
                    ),
                )
            except (RuntimeError, ValueError) as error:
                progress.empty()
                st.error(str(error))
            else:
                progress.progress(1.0, text="Attendance session saved.")
                by_id = {student["id"]: student for student in registered}
                report = pd.DataFrame([
                    {**by_id[row["student_id"]], **row} for row in records
                ])
                st.success("Attendance session completed and saved to the local database.")
                st.dataframe(report, hide_index=True, use_container_width=True)

    history = registry.recent_attendance()
    if history:
        st.divider()
        st.subheader("Recent attendance history")
        st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)

if current_page == "engagement":
    if not timeline.empty:
        tl = timeline.copy()
        tl["minute"] = tl["t_start"] / 60
        st.subheader("Session engagement")
        st.caption("Class-level signals summarized in time windows; these are not grades or individual judgments.")
        st.area_chart(tl.set_index("minute")["class_engagement"], height=320)
        left, right = st.columns(2)
        with left:
            st.subheader("Room noise")
            st.line_chart(tl.set_index("minute")["noise_db"])
        with right:
            st.subheader("Speech activity")
            st.bar_chart(tl.set_index("minute")["speech_ratio"])
        if "participants" in tl:
            st.subheader("Visible participants")
            st.line_chart(tl.set_index("minute")["participants"])
    else:
        st.info("No engagement timeline is available for this session.")

if current_page == "instructor_notes":
    st.subheader("Instructor notes")
    st.caption("Assistive, evidence-grounded prompts. Use your judgment and classroom context.")
    if not suggestions:
        st.info("No suggestions for this session.")
    for s in suggestions:
        with st.expander(f"{s['priority'].upper()} · {s['time_range']} · {s['suggestion']}"):
            st.write(s["why"])
            st.caption("Observed evidence")
            st.json(s["evidence"], expanded=False)

if current_page == "learner_view":
    st.subheader("Individual session view")
    st.caption("Individual session summary. No grades, rankings, or peer comparisons.")
    ids = [s["participant_id"] for s in students]
    if ids:
        pid = st.selectbox("Participant", ids)
        s = next(x for x in students if x["participant_id"] == pid)
        attendance_col, engagement_col = st.columns(2)
        attendance_col.metric("Attendance", s["attendance"]["status"].title())
        attendance_col.metric(
            "Visible time",
            f"{s['attendance']['presence_seconds']:.1f}s",
            f"{s['attendance']['presence_percent']:.1f}% of session",
        )
        if s["engagement"]:
            e = s["engagement"]
            engagement_col.metric("Average engagement", e["average"])
            engagement_col.metric("Consistency", e["consistency"])
            st.subheader("Personal session trend")
            st.line_chart(pd.DataFrame(e["timeline"]).set_index("t_start")["engagement"])
        else:
            st.info("There was not enough visible time to calculate an engagement trend.")
        st.subheader("Session reflections")
        for n in s["notes"]:
            st.markdown(f"- {n}")
    else:
        st.info("No individual analytics are available for this session.")
