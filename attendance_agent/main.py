"""Command-line entry point.

Examples
  python -m attendance_agent --video classroom.mp4
  python -m attendance_agent --camera 0 --max-seconds 600 --show
  python -m attendance_agent --demo
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import Config
from .pipeline import SessionPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="attendance_agent",
                                description="Offline multimodal attendance & engagement agent")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to a recorded classroom video")
    src.add_argument("--camera", type=int, help="webcam index (e.g. 0) for a live session")
    src.add_argument("--demo", action="store_true", help="run on a synthetic session (no camera needed)")
    p.add_argument("--audio", help="separate audio file (default: soundtrack of the video / live mic)")
    p.add_argument("--no-audio", action="store_true", help="vision-only analysis")
    p.add_argument("--config", help="JSON file overriding default thresholds")
    p.add_argument("--out", default="output", help="output root directory (default: output)")
    p.add_argument("--max-seconds", type=float, help="stop after this many seconds")
    p.add_argument("--show", action="store_true", help="show live annotated video window")
    return p


def print_report(res) -> None:
    s = res.summary
    print("\n=== Session summary ===")
    print(f"Duration: {s['duration_seconds']}s | detected: {s['participants_detected']} | "
          f"present: {s['participants_present']} | mean engagement: {s['mean_class_engagement']}")
    print("\n=== Attendance ===")
    for a in res.attendance:
        print(f"  {a['participant_id']:>4}  {a['status']:<8} {a['presence_seconds']:>7.1f}s "
              f"({a['presence_percent']}%)  first seen {a['first_seen']}")
    print("\n=== Suggestions for the instructor ===")
    for g in res.suggestions:
        print(f"  [{g['priority']}] {g['time_range']}  {g['suggestion']}\n        why: {g['why']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    pipe = SessionPipeline(cfg)

    if args.demo:
        from .demo import build_demo
        obs, dt, dur, audio = build_demo(cfg)
        res = pipe.analyse(obs, dt, dur, None if args.no_audio else audio)
    else:
        source = args.camera if args.camera is not None else args.video
        if args.video and not Path(args.video).exists():
            print(f"Video not found: {args.video}")
            return 2
        res = pipe.run_video(source, audio_path=args.audio, use_audio=not args.no_audio,
                             show=args.show, max_seconds=args.max_seconds)

    out = pipe.save(res, Path(args.out) / f"session_{datetime.now():%Y%m%d_%H%M%S}")
    print_report(res)
    print(f"\nResults saved to: {out}")
    print("Launch the optional dashboard with: streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
