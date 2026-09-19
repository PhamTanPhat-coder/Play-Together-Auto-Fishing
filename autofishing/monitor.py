"""Cửa sổ điều khiển và theo dõi bot câu cá."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass


@dataclass(frozen=True)
class StatusSnapshot:
    state: str = "Chưa chạy"
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
    """Giao diện Monitor chạy trên main thread."""

    def __init__(self) -> None:
        self._main_thread_id = threading.get_ident()

        self._queue: queue.Queue[
            tuple[str, object | None]
        ] = queue.Queue(maxsize=200)

        self._root: tk.Tk | None = None
        self._controller = None

        self._closed = threading.Event()

        self._snapshot = StatusSnapshot()
        self._logs: list[str] = []

        # =====================================================
        # TK VARIABLES
        # =====================================================

        self._state_var: tk.StringVar | None = None
        self._uptime_var: tk.StringVar | None = None
        self._event_var: tk.StringVar | None = None
        self._confidence_var: tk.StringVar | None = None

        self._cast_var: tk.StringVar | None = None
        self._caught_var: tk.StringVar | None = None
        self._missed_var: tk.StringVar | None = None
        self._no_detect_var: tk.StringVar | None = None
        self._bite_var: tk.StringVar | None = None

        self._auto_recovery_var: tk.BooleanVar | None = None

        self._log_list: tk.Listbox | None = None

        # Scroll
        self._canvas: tk.Canvas | None = None
        self._scroll_frame: tk.Frame | None = None
        self._scrollbar: tk.Scrollbar | None = None

    # =========================================================
    # KẾT NỐI CONTROLLER
    # =========================================================

    def set_controller(self, controller) -> None:
        self._controller = controller

    # =========================================================
    # NHẬN DỮ LIỆU
    # =========================================================

    def publish(
        self,
        snapshot: StatusSnapshot,
    ) -> None:
        self._push(
            "snapshot",
            snapshot,
        )

    def log_event(
        self,
        message: str,
    ) -> None:
        self._push(
            "log",
            str(message),
        )

    # =========================================================
    # COMPATIBILITY
    # =========================================================

    def start(self) -> None:
        """
        Giữ lại để code cũ không lỗi.

        Không tạo Tkinter thread.
        """

    def close(self) -> None:
        self._push(
            "close",
            None,
        )

    def wait_closed(self) -> None:
        self._closed.wait()

    # =========================================================
    # CHẠY GIAO DIỆN
    # =========================================================

    def show(self) -> None:
        if threading.get_ident() != self._main_thread_id:
            raise RuntimeError(
                "FishingStatusMonitor.show() phải chạy ở main thread."
            )

        if self._root is not None:
            return

        self._build_ui()

        self._root.after(
            50,
            self._poll_queue,
        )

        self._root.after(
            500,
            self._refresh_uptime,
        )

        self._root.mainloop()

        self._closed.set()
        self._root = None

    # =========================================================
    # TẠO GIAO DIỆN
    # =========================================================

    def _build_ui(self) -> None:
        root = tk.Tk()
        self._root = root

        root.title(
            "PLAY TOGETHER - TỰ ĐỘNG CÂU CÁ"
        )

        # Cửa sổ nhỏ vừa phải.
        # Nội dung bên trong dài hơn sẽ cuộn.
        root.geometry(
            "430x650"
        )

        root.minsize(
            430,
            400,
        )

        root.resizable(
            False,
            True,
        )

        root.attributes(
            "-topmost",
            True,
        )

        root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

        # =====================================================
        # BIẾN
        # =====================================================

        self._state_var = tk.StringVar(
            value="Chưa chạy"
        )

        self._uptime_var = tk.StringVar(
            value="Thời gian chạy: 0 giây"
        )

        self._event_var = tk.StringVar(
            value="Sự kiện: -"
        )

        self._confidence_var = tk.StringVar(
            value="Độ tin cậy: -"
        )

        self._cast_var = tk.StringVar(
            value="0"
        )

        self._caught_var = tk.StringVar(
            value="0"
        )

        self._missed_var = tk.StringVar(
            value="0"
        )

        self._no_detect_var = tk.StringVar(
            value="0"
        )

        self._bite_var = tk.StringVar(
            value="0"
        )

        self._auto_recovery_var = tk.BooleanVar(
            value=True
        )

        root.grid_rowconfigure(
            0,
            weight=1,
        )

        root.grid_columnconfigure(
            0,
            weight=1,
        )

        # =====================================================
        # CANVAS
        # =====================================================

        self._canvas = tk.Canvas(
            root,
            highlightthickness=0,
        )

        self._canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        # =====================================================
        # SCROLLBAR
        # =====================================================

        self._scrollbar = tk.Scrollbar(
            root,
            orient="vertical",
            command=self._canvas.yview,
        )

        self._scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        self._canvas.configure(
            yscrollcommand=self._scrollbar.set,
        )

        # =====================================================
        # FRAME BÊN TRONG CANVAS
        # =====================================================

        self._scroll_frame = tk.Frame(
            self._canvas,
            padx=10,
            pady=8,
        )

        window_id = self._canvas.create_window(
            0,
            0,
            anchor="nw",
            window=self._scroll_frame,
        )

        def on_frame_configure(_event=None):
            self._canvas.configure(
                scrollregion=self._canvas.bbox("all")
            )

        def on_canvas_configure(event):
            self._canvas.itemconfigure(
                window_id,
                width=event.width,
            )

        self._scroll_frame.bind(
            "<Configure>",
            on_frame_configure,
        )

        self._canvas.bind(
            "<Configure>",
            on_canvas_configure,
        )

        # =====================================================
        # CUỘN BẰNG CON LĂN CHUỘT
        # =====================================================

        self._canvas.bind_all(
            "<MouseWheel>",
            self._on_mousewheel,
        )

        # =====================================================
        # BUILD CONTENT
        # =====================================================

        frame = self._scroll_frame

        frame.columnconfigure(
            0,
            weight=1,
        )

        # =====================================================
        # TIÊU ĐỀ
        # =====================================================

        tk.Label(
            frame,
            text="PLAY TOGETHER",
            font=("Segoe UI", 16, "bold"),
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 1),
        )

        tk.Label(
            frame,
            text="TỰ ĐỘNG CÂU CÁ",
            font=("Segoe UI", 11),
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )

        # =====================================================
        # TRẠNG THÁI
        # =====================================================

        state_frame = tk.LabelFrame(
            frame,
            text="Trạng thái",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=8,
        )

        state_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=4,
        )

        tk.Label(
            state_frame,
            textvariable=self._state_var,
            font=("Segoe UI", 15, "bold"),
        ).pack(
            pady=(0, 3),
        )

        tk.Label(
            state_frame,
            textvariable=self._uptime_var,
            font=("Segoe UI", 9),
        ).pack()

        tk.Label(
            state_frame,
            textvariable=self._confidence_var,
            font=("Segoe UI", 9),
        ).pack(
            pady=(2, 0),
        )

        # =====================================================
        # THỐNG KÊ
        # =====================================================

        stats_frame = tk.LabelFrame(
            frame,
            text="Thống kê",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=8,
        )

        stats_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=4,
        )

        for column in range(3):
            stats_frame.columnconfigure(
                column,
                weight=1,
            )

        self._create_stat_box(
            stats_frame,
            "Số lần\nthả câu",
            self._cast_var,
            0,
            0,
        )

        self._create_stat_box(
            stats_frame,
            "Câu được",
            self._caught_var,
            0,
            1,
        )

        self._create_stat_box(
            stats_frame,
            "Câu hụt",
            self._missed_var,
            0,
            2,
        )

        self._create_stat_box(
            stats_frame,
            "Không phát hiện\ncắn câu",
            self._no_detect_var,
            1,
            0,
        )

        self._create_stat_box(
            stats_frame,
            "Phát hiện\ncắn câu",
            self._bite_var,
            1,
            1,
        )

        # =====================================================
        # THÔNG BÁO
        # =====================================================

        event_frame = tk.LabelFrame(
            frame,
            text="Thông báo",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=7,
        )

        event_frame.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=4,
        )

        tk.Label(
            event_frame,
            textvariable=self._event_var,
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=375,
        ).pack(
            fill="x",
        )

        # =====================================================
        # ĐIỀU KHIỂN
        # =====================================================

        control_frame = tk.LabelFrame(
            frame,
            text="Điều khiển",
            font=("Segoe UI", 9, "bold"),
            padx=7,
            pady=7,
        )

        control_frame.grid(
            row=5,
            column=0,
            sticky="ew",
            pady=4,
        )

        control_frame.columnconfigure(
            0,
            weight=1,
        )

        control_frame.columnconfigure(
            1,
            weight=1,
        )

        self._create_button(
            control_frame,
            "▶ BẮT ĐẦU",
            self._start,
            0,
            0,
            "#2e7d32",
        )

        self._create_button(
            control_frame,
            "Ⅱ TẠM DỪNG",
            self._toggle_pause,
            0,
            1,
            "#ef6c00",
        )

        self._create_button(
            control_frame,
            "↻ KHỞI ĐỘNG LẠI",
            self._restart,
            1,
            0,
            "#1565c0",
        )

        self._create_button(
            control_frame,
            "0 ĐẶT LẠI SỐ LIỆU",
            self._reset,
            1,
            1,
            "#6a1b9a",
        )

        self._create_button(
            control_frame,
            "■ DỪNG BOT",
            self._stop,
            2,
            0,
            "#c62828",
        )

        self._create_button(
            control_frame,
            "X THOÁT",
            self._on_close,
            2,
            1,
            "#424242",
        )

        # =====================================================
        # TỰ PHỤC HỒI
        # =====================================================

        tk.Checkbutton(
            frame,
            text="Tự động chạy lại khi bot gặp lỗi",
            variable=self._auto_recovery_var,
            command=self._toggle_auto_recovery,
            font=("Segoe UI", 9),
        ).grid(
            row=6,
            column=0,
            sticky="w",
            pady=(5, 3),
        )

        # =====================================================
        # NHẬT KÝ
        # =====================================================

        log_frame = tk.LabelFrame(
            frame,
            text="Nhật ký",
            font=("Segoe UI", 9, "bold"),
            padx=6,
            pady=6,
        )

        log_frame.grid(
            row=7,
            column=0,
            sticky="ew",
            pady=3,
        )

        scrollbar = tk.Scrollbar(
            log_frame,
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

        self._log_list = tk.Listbox(
            log_frame,
            height=5,
            font=("Consolas", 8),
            yscrollcommand=scrollbar.set,
        )

        self._log_list.pack(
            side="left",
            fill="both",
            expand=True,
        )

        scrollbar.config(
            command=self._log_list.yview,
        )

    # =========================================================
    # CUỘN
    # =========================================================

    def _on_mousewheel(
        self,
        event,
    ) -> None:

        if self._canvas is None:
            return

        self._canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units",
        )

    # =========================================================
    # Ô THỐNG KÊ
    # =========================================================

    def _create_stat_box(
        self,
        parent,
        title: str,
        value_var: tk.StringVar,
        row: int,
        column: int,
    ) -> None:

        box = tk.Frame(
            parent,
            bd=1,
            relief="solid",
            padx=4,
            pady=4,
        )

        box.grid(
            row=row,
            column=column,
            padx=3,
            pady=3,
            sticky="nsew",
        )

        tk.Label(
            box,
            text=title,
            font=("Segoe UI", 8),
            justify="center",
        ).pack()

        tk.Label(
            box,
            textvariable=value_var,
            font=("Segoe UI", 18, "bold"),
        ).pack(
            pady=(1, 0),
        )

    # =========================================================
    # NÚT
    # =========================================================

    def _create_button(
        self,
        parent,
        text: str,
        command,
        row: int,
        column: int,
        bg: str,
    ) -> None:

        tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="white",
            activebackground=bg,
            activeforeground="white",
            font=("Segoe UI", 9, "bold"),
            height=1,
            relief="flat",
            bd=0,
            cursor="hand2",
        ).grid(
            row=row,
            column=column,
            padx=3,
            pady=3,
            sticky="ew",
        )

    # =========================================================
    # BUTTON ACTIONS
    # =========================================================

    def _start(self) -> None:
        if self._controller is not None:
            self._controller.start()

    def _toggle_pause(self) -> None:
        if self._controller is not None:
            self._controller.toggle_pause()

    def _restart(self) -> None:
        if self._controller is not None:
            self._controller.restart()

    def _reset(self) -> None:
        if self._controller is not None:
            self._controller.reset()

    def _stop(self) -> None:
        if self._controller is not None:
            self._controller.stop()

    def _toggle_auto_recovery(self) -> None:
        if self._controller is None:
            return

        enabled = bool(
            self._auto_recovery_var.get()
        )

        self._controller.set_auto_recovery(
            enabled
        )

    def _on_close(self) -> None:
        if self._controller is not None:
            self._controller.shutdown()

        if self._root is not None:
            self._root.destroy()

    # =========================================================
    # QUEUE
    # =========================================================

    def _push(
        self,
        kind: str,
        payload: object | None,
    ) -> None:

        try:
            self._queue.put_nowait(
                (kind, payload)
            )

        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._queue.put_nowait(
                    (kind, payload)
                )
            except queue.Full:
                pass

    def _poll_queue(self) -> None:
        root = self._root

        if root is None:
            return

        latest_snapshot = None
        log_messages: list[str] = []
        should_close = False

        while True:
            try:
                kind, payload = (
                    self._queue.get_nowait()
                )

            except queue.Empty:
                break

            if (
                kind == "snapshot"
                and isinstance(
                    payload,
                    StatusSnapshot,
                )
            ):
                latest_snapshot = payload

            elif (
                kind == "log"
                and isinstance(
                    payload,
                    str,
                )
            ):
                log_messages.append(
                    payload
                )

            elif kind == "close":
                should_close = True

        if latest_snapshot is not None:
            self._snapshot = latest_snapshot
            self._render_snapshot(
                latest_snapshot
            )

        for message in log_messages:
            self._append_log(
                message
            )

        if should_close:
            root.destroy()
            return

        root.after(
            50,
            self._poll_queue,
        )

    # =========================================================
    # HIỂN THỊ
    # =========================================================

    def _render_snapshot(
        self,
        snap: StatusSnapshot,
    ) -> None:

        if self._state_var is not None:
            self._state_var.set(
                snap.state
            )

        if self._cast_var is not None:
            self._cast_var.set(
                str(snap.casts)
            )

        if self._caught_var is not None:
            self._caught_var.set(
                str(snap.caught)
            )

        if self._missed_var is not None:
            self._missed_var.set(
                str(snap.missed)
            )

        if self._no_detect_var is not None:
            self._no_detect_var.set(
                str(snap.no_detect)
            )

        if self._bite_var is not None:
            self._bite_var.set(
                str(snap.bites)
            )

        if self._event_var is not None:
            self._event_var.set(
                f"Sự kiện: {snap.last_event}"
            )

        if self._confidence_var is not None:
            if snap.last_conf is None:
                self._confidence_var.set(
                    "Độ tin cậy: -"
                )
            else:
                self._confidence_var.set(
                    f"Độ tin cậy: "
                    f"{snap.last_conf:.2f}"
                )

    # =========================================================
    # LOG
    # =========================================================

    def _append_log(
        self,
        message: str,
    ) -> None:

        self._logs.append(
            message
        )

        self._logs = self._logs[-8:]

        if self._log_list is None:
            return

        self._log_list.delete(
            0,
            tk.END,
        )

        for item in self._logs:
            self._log_list.insert(
                tk.END,
                item,
            )

        if self._logs:
            self._log_list.yview_moveto(
                1.0,
            )

    # =========================================================
    # UPTIME
    # =========================================================

    def _refresh_uptime(self) -> None:
        root = self._root

        if root is None:
            return

        started = self._snapshot.started_at

        if (
            started > 0
            and self._uptime_var is not None
        ):
            elapsed = max(
                0,
                int(
                    time.monotonic()
                    - started
                ),
            )

            self._uptime_var.set(
                f"Thời gian chạy: "
                f"{elapsed} giây"
            )

        root.after(
            500,
            self._refresh_uptime,
        )