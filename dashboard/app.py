"""Optional Streamlit dashboard (Activity 6.2).

Run:  streamlit run dashboard/app.py
Reads saved sessions from ./output (or generates a synthetic demo session).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attendance_agent.config import Config  # noqa: E402
from attendance_agent.demo import build_demo  # noqa: E402
from attendance_agent.pipeline import SessionPipeline  # noqa: E402

st.set_page_config(page_title="Attendance & Engagement Agent", layout="wide")
st.title("Multimodal Attendance & Engagement Agent")
st.caption("Offline and privacy-preserving. Decision support for instructors, not automated judgement.")

OUT = Path(st.sidebar.text_input("Output folder", "output"))
sessions = sorted([p for p in OUT.glob("session_*") if p.is_dir()], reverse=True) if OUT.exists() else []
choice = st.sidebar.selectbox("Session", ["(synthetic demo)"] + [p.name for p in sessions])


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

c = st.columns(4)
c[0].metric("Duration", f"{summary['duration_seconds'] / 60:.1f} min")
c[1].metric("Present", f"{summary['participants_present']} / {summary['participants_detected']}")
c[2].metric("Mean engagement", summary["mean_class_engagement"])
c[3].metric("Mean noise", f"{summary.get('mean_noise_db', 'n/a')} dBFS")

tab1, tab2, tab3, tab4 = st.tabs(["Attendance", "Engagement timeline", "Instructor suggestions", "Student analytics"])

with tab1:
    st.dataframe(att, hide_index=True)
    st.caption("Present = cumulative visible time above the configured threshold (not a single frame).")

with tab2:
    if not timeline.empty:
        tl = timeline.copy()
        tl["minute"] = tl["t_start"] / 60
        st.subheader("Class engagement over time")
        st.line_chart(tl.set_index("minute")["class_engagement"])
        left, right = st.columns(2)
        left.subheader("Ambient noise (dBFS)")
        left.line_chart(tl.set_index("minute")["noise_db"])
        right.subheader("Speech activity (share of window)")
        right.bar_chart(tl.set_index("minute")["speech_ratio"])

with tab3:
    if not suggestions:
        st.info("No suggestions for this session.")
    for s in suggestions:
        with st.expander(f"[{s['priority'].upper()}] {s['time_range']}  -  {s['suggestion']}"):
            st.write(s["why"])
            st.json(s["evidence"])
    st.caption("Suggestions are assistive hints grounded in observed trends; you decide what to do.")

with tab4:
    st.caption("Each learner sees only their own session. No grades, rankings or peer comparison.")
    ids = [s["participant_id"] for s in students]
    if ids:
        pid = st.selectbox("Participant", ids)
        s = next(x for x in students if x["participant_id"] == pid)
        st.write(s["attendance"])
        if s["engagement"]:
            e = s["engagement"]
            a, b = st.columns(2)
            a.metric("Average engagement", e["average"])
            b.metric("Consistency", e["consistency"])
            st.line_chart(pd.DataFrame(e["timeline"]).set_index("t_start")["engagement"])
        for n in s["notes"]:
            st.write("- " + n)
