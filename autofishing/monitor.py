"""Small always-on-top status window for the fishing bot."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StatusSnapshot:
    state: str = "-"
    casts: int = 0
    caught: int = 0
    missed: int = 0
    no_detect: int = 0
    bites: int = 0
    last_conf: float | None = None
    last_event: str = "-"
    started_at: float = 0.0
    fallback_store: int = 0


class FishingStatusMonitor:
    """Thread-safe status bridge + small Tkinter window.

    The bot thread only pushes immutable snapshots into a queue. All Tk calls
    happen inside the monitor's own GUI thread so the fishing loop is not
    blocked by the window.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[StatusSnapshot] = queue.Queue(maxsize=20)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run_gui,
            name="FishingStatusMonitor",
            daemon=True,
        )
        self._thread.start()

    def publish(self, snapshot: StatusSnapshot) -> None:
        if not self._started:
            return
        # Keep the newest state; an old UI update is never worth blocking the
        # fishing thread over.
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop_event.set()

    def _run_gui(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:  # pragma: no cover - Windows normally has Tk
            print(f"[monitor] Tkinter unavailable: {exc}")
            return

        try:
            root = tk.Tk()
        except Exception as exc:
            print(f"[monitor] cannot open status window: {exc}")
            return

        root.title("Play Together Auto Fishing")
        root.geometry("290x265+20+80")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        outer = tk.Frame(root, padx=12, pady=10)
        outer.pack(fill="both", expand=True)

        title = tk.Label(
            outer,
            text="PLAY TOGETHER AUTO",
            font=("Segoe UI", 12, "bold"),
        )
        title.pack(anchor="w")

        state_var = tk.StringVar(value="Trạng thái: -")
        state_label = tk.Label(
            outer,
            textvariable=state_var,
            font=("Segoe UI", 10, "bold"),
        )
        state_label.pack(anchor="w", pady=(4, 8))

        stats_frame = tk.Frame(outer)
        stats_frame.pack(fill="x")

        vars_map = {
            "casts": tk.StringVar(value="0"),
            "caught": tk.StringVar(value="0"),
            "missed": tk.StringVar(value="0"),
            "bites": tk.StringVar(value="0"),
            "no_detect": tk.StringVar(value="0"),
            "confidence": tk.StringVar(value="-"),
            "rate": tk.StringVar(value="0.0%"),
            "fallback": tk.StringVar(value="0"),
        }

        rows = [
            ("Lần thả", "casts"),
            ("Cá bắt được", "caught"),
            ("Cá hụt", "missed"),
            ("Phát hiện !", "bites"),
            ("Không thấy ! >45s", "no_detect"),
            ("Conf ! gần nhất", "confidence"),
            ("Store fallback", "fallback"),
            ("Tỷ lệ bắt", "rate"),
        ]

        for row, (label_text, key) in enumerate(rows):
            tk.Label(stats_frame, text=label_text, anchor="w").grid(
                row=row,
                column=0,
                sticky="w",
                pady=1,
            )
            tk.Label(
                stats_frame,
                textvariable=vars_map[key],
                anchor="e",
                font=("Segoe UI", 9, "bold"),
            ).grid(row=row, column=1, sticky="e", pady=1)

        stats_frame.columnconfigure(1, weight=1)

        event_var = tk.StringVar(value="Sự kiện: -")
        event_label = tk.Label(
            outer,
            textvariable=event_var,
            anchor="w",
            justify="left",
            wraplength=255,
        )
        event_label.pack(fill="x", pady=(8, 0))

        uptime_var = tk.StringVar(value="Thời gian: 00:00:00")
        uptime_label = tk.Label(outer, textvariable=uptime_var, anchor="w")
        uptime_label.pack(fill="x", pady=(4, 0))

        latest = StatusSnapshot(started_at=time.monotonic())

        def format_uptime(started_at: float) -> str:
            elapsed = max(0, int(time.monotonic() - started_at))
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def apply_snapshot(snapshot: StatusSnapshot) -> None:
            nonlocal latest
            latest = snapshot
            state_var.set(f"Trạng thái: {snapshot.state}")
            vars_map["casts"].set(str(snapshot.casts))
            vars_map["caught"].set(str(snapshot.caught))
            vars_map["missed"].set(str(snapshot.missed))
            vars_map["bites"].set(str(snapshot.bites))
            vars_map["no_detect"].set(str(snapshot.no_detect))
            vars_map["fallback"].set(str(snapshot.fallback_store))
            vars_map["confidence"].set(
                "-" if snapshot.last_conf is None else f"{snapshot.last_conf:.2f}"
            )
            total_finished = snapshot.caught + snapshot.missed
            rate = (
                snapshot.caught / total_finished * 100
                if total_finished > 0
                else 0.0
            )
            vars_map["rate"].set(f"{rate:.1f}%")
            event_var.set(f"Sự kiện: {snapshot.last_event}")

        def poll() -> None:
            nonlocal latest
            if self._stop_event.is_set():
                try:
                    root.destroy()
                except Exception:
                    pass
                return

            newest: StatusSnapshot | None = None
            while True:
                try:
                    newest = self._queue.get_nowait()
                except queue.Empty:
                    break
            if newest is not None:
                apply_snapshot(newest)

            uptime_var.set(
                f"Thời gian: {format_uptime(latest.started_at)}"
            )
            root.after(150, poll)

        root.after(50, poll)
        root.mainloop()
