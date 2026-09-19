"""Fishing state machine."""

from __future__ import annotations

import ctypes
import enum
import os
import queue
import tempfile
import threading
import time
import tkinter as tk

import cv2

from autofishing.capture.adb import AdbDevice
from autofishing.capture.bluestacks import BlueStacksCapture
from autofishing.capture.screen import ScreenCapture
from autofishing.config import BotSettings
from autofishing.constants import UI_CLASSES
from autofishing.detection.detector import Detection, YoloDetector
from autofishing.input.clickers import (
    BlueStacksClicker,
    ScreenClicker,
)
from autofishing.monitor import (
    FishingStatusMonitor,
    StatusSnapshot,
)
from autofishing.protocols import (
    Clicker,
    FrameSource,
)


class BotState(enum.Enum):
    CAST = "cast"
    WAITING_BITE = "waiting_bite"
    AFTER_CATCH = "after_catch"
    REPAIR = "repair"
    COOLDOWN = "cooldown"


_STATE_LABELS = {
    BotState.CAST: "ĐANG THẢ CÂU",
    BotState.WAITING_BITE: "CHỜ CÁ CẮN",
    BotState.AFTER_CATCH: "ĐANG KIỂM TRA CÁ",
    BotState.REPAIR: "ĐANG SỬA CẦN",
    BotState.COOLDOWN: "CHỜ THẢ LẠI",
}


class _BiteDebugWindow:
    """Cửa sổ debug hiển thị ROI mà YOLO đang nhìn."""

    WINDOW_TITLE = "BITE ROI DEBUG"
    WINDOW_WIDTH = 420
    WINDOW_HEIGHT = 252

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue(
            maxsize=1
        )

        self._thread = threading.Thread(
            target=self._run,
            name="bite-debug-window",
            daemon=True,
        )

        self._root: tk.Tk | None = None
        self._label: tk.Label | None = None
        self._hwnd: int | None = None

        self._started = threading.Event()
        self._stop_event = threading.Event()

        self._closed_by_user = False
        self._started_once = False

        self._last_update_at = 0.0
        self._update_interval = 0.10

        self._temp_dir = tempfile.mkdtemp(
            prefix="bite_roi_debug_"
        )

        self._image_error_reported = False

    # =========================================================
    # START
    # =========================================================

    def start(self) -> None:
        if self._started_once:
            return

        self._started_once = True

        self._thread.start()

        self._started.wait(
            timeout=3.0
        )

    # =========================================================
    # UPDATE IMAGE
    # =========================================================

    def update(
        self,
        frame,
        bite: Detection | None,
    ) -> None:
        if self._closed_by_user:
            return

        if self._stop_event.is_set():
            return

        now = time.monotonic()

        if (
            now - self._last_update_at
            < self._update_interval
        ):
            return

        self._last_update_at = now

        debug_frame = frame.copy()

        h, w = debug_frame.shape[:2]

        cv2.rectangle(
            debug_frame,
            (0, 0),
            (w - 1, h - 1),
            (0, 255, 0),
            2,
        )

        center_x = w // 2
        center_y = h // 2

        cv2.line(
            debug_frame,
            (center_x, 0),
            (center_x, h - 1),
            (0, 255, 255),
            1,
        )

        cv2.line(
            debug_frame,
            (0, center_y),
            (w - 1, center_y),
            (0, 255, 255),
            1,
        )

        cv2.putText(
            debug_frame,
            f"ROI: {w} x {h}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if bite is not None:
            cv2.rectangle(
                debug_frame,
                (
                    bite.x1,
                    bite.y1,
                ),
                (
                    bite.x2,
                    bite.y2,
                ),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                debug_frame,
                f"! conf={bite.confidence:.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                debug_frame,
                f"center={bite.center}",
                (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        else:
            cv2.putText(
                debug_frame,
                "NO ! DETECTED",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        display = cv2.resize(
            debug_frame,
            (
                self.WINDOW_WIDTH,
                self.WINDOW_HEIGHT,
            ),
            interpolation=cv2.INTER_AREA,
        )

        ok, encoded = cv2.imencode(
            ".png",
            display,
        )

        if not ok:
            return

        png_bytes = encoded.tobytes()

        # -----------------------------------------------------
        # GHI FILE PNG TẠM
        # -----------------------------------------------------

        try:
            fd, file_path = tempfile.mkstemp(
                suffix=".png",
                prefix="frame_",
                dir=self._temp_dir,
            )

            os.close(fd)

            with open(
                file_path,
                "wb",
            ) as file:
                file.write(
                    png_bytes
                )

        except Exception as exc:
            if not self._image_error_reported:
                print(
                    "[debug-window] "
                    f"PNG write error: {exc}"
                )

                self._image_error_reported = True

            return

        # -----------------------------------------------------
        # CHỈ GIỮ FRAME MỚI NHẤT
        # -----------------------------------------------------

        try:
            while True:
                old_path = (
                    self._queue.get_nowait()
                )

                try:
                    os.remove(
                        old_path
                    )
                except OSError:
                    pass

        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(
                file_path
            )

        except queue.Full:
            try:
                os.remove(
                    file_path
                )
            except OSError:
                pass

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self) -> None:
        """
        Chỉ set Event.

        Không gọi root.after() hoặc bất kỳ Tkinter API nào
        từ thread bot.

        Chính thread Tkinter sẽ tự phát hiện Event và destroy.
        """

        self._stop_event.set()

    # =========================================================
    # TK THREAD
    # =========================================================

    def _run(self) -> None:
        try:
            root = tk.Tk()

            self._root = root

            root.title(
                self.WINDOW_TITLE
            )

            root.geometry(
                f"{self.WINDOW_WIDTH}x"
                f"{self.WINDOW_HEIGHT}"
                f"+10+50"
            )

            root.resizable(
                False,
                False,
            )

            root.attributes(
                "-topmost",
                True,
            )

            label = tk.Label(
                root,
                bd=0,
                highlightthickness=0,
                bg="black",
                text="Đang chờ frame...",
                fg="white",
            )

            label.pack(
                fill="both",
                expand=True,
            )

            self._label = label

            root.protocol(
                "WM_DELETE_WINDOW",
                self._on_user_close,
            )

            root.update_idletasks()

            self._hwnd = root.winfo_id()

            self._make_topmost()
            self._show_without_activate()

            self._started.set()

            root.after(
                50,
                self._pump_frame,
            )

            root.after(
                200,
                self._keep_topmost,
            )

            root.mainloop()

        except Exception as exc:
            print(
                f"[debug-window] ERROR: {exc}"
            )

        finally:
            self._started.set()

            self._cleanup_temp_files()

            self._root = None
            self._label = None
            self._hwnd = None

    # =========================================================
    # PUMP FRAME
    # =========================================================

    def _pump_frame(self) -> None:
        root = self._root

        if root is None:
            return

        # -----------------------------------------------------
        # Bot thread chỉ set Event.
        # Tk thread tự destroy ở đây.
        # -----------------------------------------------------

        if self._stop_event.is_set():
            try:
                root.destroy()
            except Exception:
                pass

            return

        latest_path: str | None = None

        try:
            while True:
                latest_path = (
                    self._queue.get_nowait()
                )

        except queue.Empty:
            pass

        if latest_path is not None:
            try:
                photo = tk.PhotoImage(
                    master=root,
                    file=latest_path,
                )

                if self._label is not None:
                    self._label.configure(
                        image=photo,
                        text="",
                    )

                    self._label.image = photo

                self._image_error_reported = False

            except Exception as exc:
                if not self._image_error_reported:
                    print(
                        "[debug-window] "
                        f"image error: {exc}"
                    )

                    self._image_error_reported = True

            finally:
                try:
                    os.remove(
                        latest_path
                    )
                except OSError:
                    pass

        root.after(
            50,
            self._pump_frame,
        )

    # =========================================================
    # KEEP TOPMOST
    # =========================================================

    def _keep_topmost(self) -> None:
        root = self._root

        if root is None:
            return

        if self._stop_event.is_set():
            try:
                root.destroy()
            except Exception:
                pass

            return

        self._make_topmost()
        self._show_without_activate()

        root.after(
            200,
            self._keep_topmost,
        )

    # =========================================================
    # WINDOWS TOPMOST
    # =========================================================

    def _make_topmost(self) -> None:
        if not self._hwnd:
            return

        try:
            HWND_TOPMOST = -1

            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOACTIVATE = 0x0010
            SWP_SHOWWINDOW = 0x0040

            ctypes.windll.user32.SetWindowPos(
                self._hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                (
                    SWP_NOSIZE
                    | SWP_NOMOVE
                    | SWP_NOACTIVATE
                    | SWP_SHOWWINDOW
                ),
            )

        except Exception:
            pass

    def _show_without_activate(self) -> None:
        if not self._hwnd:
            return

        try:
            SW_SHOWNOACTIVATE = 4

            ctypes.windll.user32.ShowWindow(
                self._hwnd,
                SW_SHOWNOACTIVATE,
            )

        except Exception:
            pass

    # =========================================================
    # USER CLOSE
    # =========================================================

    def _on_user_close(self) -> None:
        self._closed_by_user = True

        self._stop_event.set()

        root = self._root

        if root is not None:
            root.destroy()

    # =========================================================
    # CLEAN TEMP FILES
    # =========================================================

    def _cleanup_temp_files(self) -> None:
        try:
            for name in os.listdir(
                self._temp_dir
            ):
                path = os.path.join(
                    self._temp_dir,
                    name,
                )

                try:
                    os.remove(path)
                except OSError:
                    pass

            try:
                os.rmdir(
                    self._temp_dir
                )
            except OSError:
                pass

        except Exception:
            pass


class FishingBot:
    def __init__(
        self,
        capture: FrameSource,
        detector: YoloDetector,
        config: BotSettings,
        clicker: Clicker | None = None,
        monitor: FishingStatusMonitor | None = None,
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.config = config

        self.clicker = (
            clicker
            or ScreenClicker()
        )

        # =====================================================
        # MONITOR OWNERSHIP
        # =====================================================
        #
        # Nếu BotController truyền Monitor vào:
        #
        #     monitor=monitor
        #
        # thì FishingBot KHÔNG được đóng Monitor.
        #
        # Điều này rất quan trọng khi RESTART.
        #
        # Bot cũ chết -> Monitor vẫn còn -> Bot mới dùng lại.
        #

        if monitor is None:
            self._monitor = (
                FishingStatusMonitor()
            )

            self._owns_monitor = True

        else:
            self._monitor = monitor
            self._owns_monitor = False

        self.state = (
            BotState.CAST
            if config.auto_cast
            else BotState.WAITING_BITE
        )

        self._state_entered_at = (
            time.monotonic()
        )

        self._running = False

        self._paused = False
        self._pause_started_at: float | None = None

        self._last_repair_click_at = 0.0
        self._repair_opened_bag = False
        self._too_late_seen_at: float | None = None

        # =====================================================
        # THỐNG KÊ
        # =====================================================

        self._session_started_at = (
            time.monotonic()
        )

        self._casts = 0
        self._caught = 0
        self._missed = 0
        self._no_detect = 0
        self._bites = 0
        self._fallback_store = 0

        self._last_conf: float | None = None
        self._last_event = "Khởi động"

        self._miss_recorded_for_cast = False

        # =====================================================
        # DEBUG WINDOW
        # =====================================================

        self._debug_window_enabled = True

        self._debug_window = (
            _BiteDebugWindow()
        )

    # =========================================================
    # CONTROL
    # =========================================================

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> bool:
        if not self._running:
            return False

        if self._paused:
            return True

        self._paused = True

        self._pause_started_at = (
            time.monotonic()
        )

        self._event(
            "ĐÃ TẠM DỪNG BOT"
        )

        return True

    def resume(self) -> bool:
        if not self._running:
            return False

        if not self._paused:
            return False

        now = time.monotonic()

        if self._pause_started_at is not None:
            paused_duration = (
                now
                - self._pause_started_at
            )

            self._state_entered_at += (
                paused_duration
            )

        self._paused = False
        self._pause_started_at = None

        self._event(
            "ĐÃ TIẾP TỤC BOT"
        )

        return True

    def toggle_pause(self) -> bool:
        if self._paused:
            return self.resume()

        return self.pause()

    def reset_stats(self) -> None:
        self._casts = 0
        self._caught = 0
        self._missed = 0
        self._no_detect = 0
        self._bites = 0
        self._fallback_store = 0

        self._last_conf = None

        self._session_started_at = (
            time.monotonic()
        )

        self._miss_recorded_for_cast = False

        self._event(
            "ĐÃ ĐẶT LẠI SỐ LIỆU"
        )

    def export_statistics(
        self,
    ) -> dict[str, float | int | None]:
        return {
            "casts": self._casts,
            "caught": self._caught,
            "missed": self._missed,
            "no_detect": self._no_detect,
            "bites": self._bites,
            "fallback_store": self._fallback_store,
            "last_conf": self._last_conf,
            "started_at": self._session_started_at,
        }

    def restore_statistics(
        self,
        stats: dict[str, float | int | None],
    ) -> None:
        self._casts = int(
            stats.get(
                "casts",
                0,
            )
            or 0
        )

        self._caught = int(
            stats.get(
                "caught",
                0,
            )
            or 0
        )

        self._missed = int(
            stats.get(
                "missed",
                0,
            )
            or 0
        )

        self._no_detect = int(
            stats.get(
                "no_detect",
                0,
            )
            or 0
        )

        self._bites = int(
            stats.get(
                "bites",
                0,
            )
            or 0
        )

        self._fallback_store = int(
            stats.get(
                "fallback_store",
                0,
            )
            or 0
        )

        last_conf = stats.get(
            "last_conf"
        )

        self._last_conf = (
            float(last_conf)
            if last_conf is not None
            else None
        )

        started_at = stats.get(
            "started_at"
        )

        if started_at is not None:
            self._session_started_at = float(
                started_at
            )

    def stop(self) -> None:
        """
        Dừng bot hiện tại.

        Không đóng Monitor ở đây.
        Monitor được BotController dùng chung
        và phải sống qua thao tác RESTART.
        """

        self._running = False
        self._paused = False
        self._pause_started_at = None

    # =========================================================
    # STATUS
    # =========================================================

    def _state_label(self) -> str:
        label = _STATE_LABELS.get(
            self.state,
            self.state.value,
        )

        if self._paused:
            return (
                f"TẠM DỪNG - {label}"
            )

        return label

    def _publish_status(self) -> None:
        self._monitor.publish(
            StatusSnapshot(
                state=self._state_label(),
                casts=self._casts,
                caught=self._caught,
                missed=self._missed,
                no_detect=self._no_detect,
                bites=self._bites,
                last_conf=self._last_conf,
                last_event=self._last_event,
                started_at=self._session_started_at,
                fallback_store=self._fallback_store,
            )
        )

    # =========================================================
    # EVENT
    # =========================================================

    def _event(
        self,
        message: str,
    ) -> None:
        self._last_event = message

        print(
            f"[bot] {message}"
        )

        self._publish_status()

        try:
            self._monitor.log_event(
                message
            )
        except Exception:
            pass

    # =========================================================
    # RECORD MISS
    # =========================================================

    def _record_miss(
        self,
        reason: str,
    ) -> bool:
        if self._miss_recorded_for_cast:
            return False

        self._miss_recorded_for_cast = True

        self._missed += 1

        self._event(
            f"HỤT #{self._missed}: {reason}"
        )

        return True

    # =========================================================
    # ELAPSED
    # =========================================================

    def _elapsed_ms(self) -> float:
        return (
            time.monotonic()
            - self._state_entered_at
        ) * 1000

    # =========================================================
    # SET STATE
    # =========================================================

    def _set_state(
        self,
        state: BotState,
    ) -> None:
        self.state = state

        self._state_entered_at = (
            time.monotonic()
        )

        self._too_late_seen_at = None

        if state == BotState.REPAIR:
            self._repair_opened_bag = False
            self._last_repair_click_at = 0.0

        if state == BotState.WAITING_BITE:
            self._last_ui_scan = -1
            self._last_hb = -1

        elif state == BotState.AFTER_CATCH:
            self._last_ac = -1

        elif state == BotState.REPAIR:
            self._last_rp = -1
            self._last_bag_warn = -1

        print(
            f"[bot] state -> {state.value}"
        )

        self._publish_status()

    # =========================================================
    # ACTION
    # =========================================================

    def _do(
        self,
        mode: str,
        label: str,
        x: int | None = None,
        y: int | None = None,
        *,
        refocus: bool = False,
    ) -> None:
        if self.config.click_delay_ms:
            time.sleep(
                self.config.click_delay_ms
                / 1000
            )

        if mode in (
            "space",
            "key",
        ):
            self.clicker.press_key(
                self.config.action_key
            )

            print(
                f"[bot] {label} via key "
                f"'{self.config.action_key}'"
            )

        else:
            assert (
                x is not None
                and y is not None
            )

            if isinstance(
                self.clicker,
                BlueStacksClicker,
            ):
                self.clicker.click(
                    x,
                    y,
                    refocus=refocus,
                )

            else:
                self.clicker.click(
                    x,
                    y,
                )

            print(
                f"[bot] {label} click "
                f"({x}, {y})"
            )

    # =========================================================
    # FRAME TO LOGICAL
    # =========================================================

    def _frame_to_logical(
        self,
        frame,
        x: int,
        y: int,
        *,
        from_roi: bool = False,
    ) -> tuple[int, int]:
        fh, fw = frame.shape[:2]

        if isinstance(
            self.clicker,
            BlueStacksClicker,
        ):
            if (
                from_roi
                and isinstance(
                    self.capture,
                    BlueStacksCapture,
                )
                and self.capture.roi_width
                and self.capture.roi_height
            ):
                return (
                    self.capture.roi_left
                    + int(
                        x
                        * self.capture.roi_width
                        / fw
                    ),
                    self.capture.roi_top
                    + int(
                        y
                        * self.capture.roi_height
                        / fh
                    ),
                )

            return (
                int(
                    x
                    * self.clicker.image_width
                    / fw
                ),
                int(
                    y
                    * self.clicker.image_height
                    / fh
                ),
            )

        if (
            from_roi
            and isinstance(
                self.capture,
                AdbDevice,
            )
            and self.capture.region.width
        ):
            return (
                x
                + self.capture.region.left,
                y
                + self.capture.region.top,
            )

        if (
            from_roi
            and isinstance(
                self.capture,
                ScreenCapture,
            )
        ):
            return (
                x
                + self.capture.region.left,
                y
                + self.capture.region.top,
            )

        return x, y

    # =========================================================
    # GRAB
    # =========================================================

    def _grab(
        self,
        *,
        full: bool = False,
    ):
        grab = getattr(
            self.capture,
            "grab_bgr",
            None,
        )

        if grab is None:
            raise RuntimeError(
                "capture missing grab_bgr"
            )

        try:
            return grab(
                full=full
            )

        except TypeError:
            return grab()

    # =========================================================
    # UI DETECTION
    # =========================================================

    def _detect_ui(
        self,
        frame,
    ) -> dict[str, Detection]:
        return self.detector.find_any(
            frame,
            UI_CLASSES,
            imgsz=self.config.ui_imgsz,
            conf=self.config.ui_confidence,
        )

    # =========================================================
    # CLICK DETECTION
    # =========================================================

    def _click_det(
        self,
        frame,
        det: Detection,
        label: str,
        mode: str | None = None,
    ) -> None:
        x, y = self._frame_to_logical(
            frame,
            *det.center,
            from_roi=False,
        )

        print(
            f"[bot] {label} "
            f"conf={det.confidence:.2f}"
        )

        self._do(
            mode
            or self.config.action_mode,
            label,
            x,
            y,
            refocus=False,
        )

    # =========================================================
    # REPAIR
    # =========================================================

    def _can_repair_click(self) -> bool:
        gap = (
            self.config.repair_click_gap_ms
            / 1000
        )

        return (
            time.monotonic()
            - self._last_repair_click_at
        ) >= gap

    def _mark_repair_click(self) -> None:
        self._last_repair_click_at = (
            time.monotonic()
        )

    def _ui_requests_repair(
        self,
        ui: dict[str, Detection],
    ) -> bool:
        return any(
            key in ui
            for key in (
                "tool-is-broken",
                "paid-repair-button",
                "paid-repair-text",
                "repair-button",
                "repair-done-button",
            )
        )

    # =========================================================
    # DEBUG
    # =========================================================

    def _show_bite_debug(
        self,
        frame,
        bite: Detection | None,
    ) -> None:
        if not self._debug_window_enabled:
            return

        self._debug_window.update(
            frame,
            bite,
        )

    # =========================================================
    # CAST
    # =========================================================

    def _tick_cast(self) -> None:
        if (
            self._elapsed_ms()
            < self.config.cast_delay_ms
        ):
            return

        frame = self._grab(
            full=True
        )

        ui = self._detect_ui(
            frame
        )

        if self._ui_requests_repair(
            ui
        ):
            self._event(
                "Phát hiện giao diện sửa cần → sửa"
            )

            self._set_state(
                BotState.REPAIR
            )

            return

        self._miss_recorded_for_cast = False

        self._do(
            self.config.cast_mode,
            "cast",
            self.config.cast_x,
            self.config.cast_y,
            refocus=True,
        )

        self._casts += 1

        self._event(
            f"Đã thả câu #{self._casts}"
        )

        self._set_state(
            BotState.WAITING_BITE
        )

    # =========================================================
    # COOLDOWN
    # =========================================================

    def _tick_cooldown(self) -> None:
        if (
            self._elapsed_ms()
            < self.config.cooldown_after_bite_ms
        ):
            return

        if self.config.auto_cast:
            self._set_state(
                BotState.CAST
            )

        else:
            self._set_state(
                BotState.WAITING_BITE
            )

    # =========================================================
    # WAITING BITE
    # =========================================================

    def _tick_waiting_bite(self) -> None:
        if self._elapsed_ms() >= 45000:
            self._no_detect += 1

            self._event(
                "Không thấy ! >45s → "
                "bắt đầu lại vòng câu"
            )

            self._set_state(
                BotState.COOLDOWN
            )

            return

        if (
            self._elapsed_ms()
            < self.config.min_wait_before_bite_ms
        ):
            return

        current_ui_scan = int(
            self._elapsed_ms()
            / 1500
        )

        if (
            current_ui_scan
            != getattr(
                self,
                "_last_ui_scan",
                -1,
            )
        ):
            self._last_ui_scan = (
                current_ui_scan
            )

            ui_frame = self._grab(
                full=True
            )

            ui = self._detect_ui(
                ui_frame
            )

            if self._ui_requests_repair(
                ui
            ):
                self._event(
                    "Phát hiện giao diện sửa cần "
                    "khi đang chờ cá"
                )

                self._set_state(
                    BotState.REPAIR
                )

                return

            if (
                "too-late-run-away"
                in ui
            ):
                self._record_miss(
                    "game báo too-late "
                    "khi đang chờ cá"
                )

                self._set_state(
                    BotState.COOLDOWN
                )

                return

            if (
                "store-button"
                in ui
            ):
                self._caught += 1

                self._event(
                    "Bắt được cá "
                    "(Store còn sót lại)"
                )

                self._click_det(
                    ui_frame,
                    ui["store-button"],
                    "Store",
                    self.config.store_mode,
                )

                self._set_state(
                    BotState.COOLDOWN
                )

                return

        # -----------------------------------------------------
        # YOLO BITE DETECTION
        # -----------------------------------------------------

        t0 = time.perf_counter()

        frame = self._grab(
            full=False
        )

        t1 = time.perf_counter()

        bite = self.detector.find_bite(
            frame
        )

        t2 = time.perf_counter()

        grab_ms = (
            t1 - t0
        ) * 1000

        infer_ms = (
            t2 - t1
        ) * 1000

        self._show_bite_debug(
            frame,
            bite,
        )

        if bite is None:
            heartbeat = int(
                self._elapsed_ms()
                / 3000
            )

            if (
                heartbeat
                != getattr(
                    self,
                    "_last_hb",
                    -1,
                )
            ):
                self._last_hb = heartbeat

                self._event(
                    f"Đang quét ! "
                    f"({self._elapsed_ms()/1000:.0f}s) "
                    f"grab={grab_ms:.0f}ms "
                    f"infer={infer_ms:.0f}ms"
                )

            return

        # -----------------------------------------------------
        # CÓ CẮN
        # -----------------------------------------------------

        self._bites += 1

        self._last_conf = (
            bite.confidence
        )

        self._event(
            f"PHÁT HIỆN ! "
            f"#{self._bites} "
            f"conf={bite.confidence:.2f} "
            f"center={bite.center} "
            f"grab={grab_ms:.0f}ms "
            f"infer={infer_ms:.0f}ms"
        )

        if self.config.click_on_detection:
            cx, cy = (
                self._frame_to_logical(
                    frame,
                    *bite.center,
                    from_roi=True,
                )
            )

            self._do(
                self.config.jerk_mode,
                "reel",
                cx,
                cy,
                refocus=False,
            )

        else:
            self._do(
                self.config.jerk_mode,
                "reel",
                self.config.reel_x,
                self.config.reel_y,
                refocus=False,
            )

        self._set_state(
            BotState.AFTER_CATCH
        )

    # =========================================================
    # AFTER CATCH
    # =========================================================

    def _tick_after_catch(self) -> None:
        if (
            self._elapsed_ms()
            < self.config.after_catch_min_ms
        ):
            return

        frame = self._grab(
            full=True
        )

        ui = self._detect_ui(
            frame
        )

        # -----------------------------------------------------
        # BẮT ĐƯỢC CÁ
        # -----------------------------------------------------

        if (
            "store-button"
            in ui
        ):
            self._caught += 1

            self._event(
                "ĐÃ BẮT ĐƯỢC CÁ → Store"
            )

            self._click_det(
                frame,
                ui["store-button"],
                "Store",
                self.config.store_mode,
            )

            self._set_state(
                BotState.COOLDOWN
            )

            return

        # -----------------------------------------------------
        # CẦN SỬA
        # -----------------------------------------------------

        if self._ui_requests_repair(
            ui
        ):
            self._event(
                "Phát hiện giao diện sửa cần "
                "sau khi reel"
            )

            self._set_state(
                BotState.REPAIR
            )

            return

        # -----------------------------------------------------
        # TOO LATE
        # -----------------------------------------------------

        if (
            "too-late-run-away"
            in ui
        ):
            self._record_miss(
                "game báo too-late sau khi reel"
            )

            if self._too_late_seen_at is None:
                self._too_late_seen_at = (
                    time.monotonic()
                )

            elif (
                time.monotonic()
                - self._too_late_seen_at
            ) * 1000 >= 1200:
                self._set_state(
                    BotState.COOLDOWN
                )

            return

        # -----------------------------------------------------
        # TIMEOUT
        # -----------------------------------------------------

        if (
            self._elapsed_ms()
            >= self.config.after_catch_timeout_ms
        ):
            self._record_miss(
                "không xác nhận được cá "
                f"sau "
                f"{self.config.after_catch_timeout_ms / 1000:.0f}s"
            )

            self._fallback_store += 1

            self._event(
                f"Store dự phòng "
                f"({self.config.store_x},"
                f"{self.config.store_y})"
            )

            self._do(
                self.config.store_mode,
                "Store(fallback)",
                self.config.store_x,
                self.config.store_y,
                refocus=False,
            )

            self._set_state(
                BotState.COOLDOWN
            )

            return

        heartbeat = int(
            self._elapsed_ms()
            / 1000
        )

        if (
            heartbeat
            != getattr(
                self,
                "_last_ac",
                -1,
            )
        ):
            self._last_ac = heartbeat

            keys = ",".join(
                ui.keys()
            ) or "-"

            self._event(
                f"Đang chờ popup... "
                f"{self._elapsed_ms()/1000:.0f}s "
                f"ui=[{keys}]"
            )

    # =========================================================
    # REPAIR
    # =========================================================

    def _tick_repair(self) -> None:
        if (
            self._elapsed_ms()
            >= self.config.repair_timeout_ms
        ):
            self._event(
                "Sửa cần quá thời gian → "
                "bắt đầu lại"
            )

            self._set_state(
                BotState.COOLDOWN
            )

            return

        frame = self._grab(
            full=True
        )

        ui = self._detect_ui(
            frame
        )

        # -----------------------------------------------------
        # ĐÃ SỬA XONG
        # -----------------------------------------------------

        if (
            "repair-done-button"
            in ui
            and self._can_repair_click()
        ):
            self._click_det(
                frame,
                ui["repair-done-button"],
                "repair-done",
            )

            self._mark_repair_click()

            time.sleep(0.4)

            self.clicker.press_key(
                "esc"
            )

            self._set_state(
                BotState.COOLDOWN
            )

            return

        # -----------------------------------------------------
        # PAID REPAIR
        # -----------------------------------------------------

        if (
            "paid-repair-button"
            in ui
            and self._can_repair_click()
        ):
            if not self.config.allow_paid_repair:
                self._event(
                    "Đã chặn sửa bằng tiền"
                )

                self._set_state(
                    BotState.COOLDOWN
                )

                return

            self._click_det(
                frame,
                ui["paid-repair-button"],
                "paid-repair",
            )

            self._mark_repair_click()

            return

        # -----------------------------------------------------
        # NÚT SỬA
        # -----------------------------------------------------

        if (
            "repair-button"
            in ui
            and self._can_repair_click()
        ):
            self._click_det(
                frame,
                ui["repair-button"],
                "repair-button",
            )

            self._mark_repair_click()

            return

        # -----------------------------------------------------
        # TAB DỤNG CỤ
        # -----------------------------------------------------

        if (
            "tool-tab"
            in ui
            and self._can_repair_click()
        ):
            if "repair-button" not in ui:
                self._click_det(
                    frame,
                    ui["tool-tab"],
                    "tool-tab",
                )

                self._mark_repair_click()

            return

        # -----------------------------------------------------
        # MỞ BALO
        # -----------------------------------------------------

        if (
            not self._repair_opened_bag
            and self._can_repair_click()
        ):
            if (
                self.config.bag_x is not None
                and self.config.bag_y is not None
            ):
                self._do(
                    "tap",
                    "open bag",
                    self.config.bag_x,
                    self.config.bag_y,
                    refocus=False,
                )

                self._repair_opened_bag = True

                self._mark_repair_click()

            elif (
                int(
                    self._elapsed_ms()
                    / 2000
                )
                != getattr(
                    self,
                    "_last_bag_warn",
                    -1,
                )
            ):
                self._last_bag_warn = int(
                    self._elapsed_ms()
                    / 2000
                )

                self._event(
                    "Chưa có bag_x/bag_y "
                    "trong config.yaml"
                )

            return

        heartbeat = int(
            self._elapsed_ms()
            / 2000
        )

        if (
            heartbeat
            != getattr(
                self,
                "_last_rp",
                -1,
            )
        ):
            self._last_rp = heartbeat

            keys = ",".join(
                ui.keys()
            ) or "-"

            self._event(
                f"Đang sửa... "
                f"{self._elapsed_ms()/1000:.0f}s "
                f"ui=[{keys}]"
            )

    # =========================================================
    # TICK
    # =========================================================

    def tick(self) -> None:
        if self._paused:
            time.sleep(0.05)
            return

        if self.state == BotState.CAST:
            self._tick_cast()

        elif self.state == BotState.COOLDOWN:
            self._tick_cooldown()

        elif self.state == BotState.AFTER_CATCH:
            self._tick_after_catch()

        elif self.state == BotState.REPAIR:
            self._tick_repair()

        else:
            self._tick_waiting_bite()

    # =========================================================
    # RUN
    # =========================================================

    def run(self) -> None:
        self._running = True

        # Monitor có thể là Monitor dùng chung từ Controller.
        # start() hiện không tạo thread Tk nữa nên an toàn.
        self._monitor.start()

        self._publish_status()

        # -----------------------------------------------------
        # DEBUG WINDOW
        # -----------------------------------------------------

        if self._debug_window_enabled:
            try:
                self._debug_window.start()

                initial_frame = self._grab(
                    full=False
                )

                self._show_bite_debug(
                    initial_frame,
                    None,
                )

            except Exception as exc:
                print(
                    "[bot] debug window init failed: "
                    f"{exc}"
                )

        print(
            "[bot] flow: cast → ! → reel "
            "→ (Store | too-late | repair) "
            "→ cooldown → loop"
        )

        print(
            "[bot] Ctrl+C to quit"
        )

        try:
            while self._running:
                self.tick()

                time.sleep(
                    self.config.poll_interval_ms
                    / 1000
                )

        except KeyboardInterrupt:
            print(
                "\n[bot] stopped"
            )

        finally:
            # =================================================
            # QUAN TRỌNG NHẤT
            # =================================================
            #
            # Bot hiện tại chỉ dọn dẹp tài nguyên của chính nó.
            #
            # Monitor KHÔNG được đóng nếu nó thuộc Controller.
            #
            # Vì vậy:
            #
            # RESTART:
            # Bot #1 -> dừng
            # Bot #1 -> đóng capture
            # Bot #1 -> đóng debug window
            # Monitor -> vẫn sống
            # Bot #2 -> được tạo
            #
            # =================================================

            self._running = False
            self._paused = False
            self._pause_started_at = None

            # -------------------------------------------------
            # Đóng capture
            # -------------------------------------------------

            try:
                self.capture.close()

            except Exception as exc:
                print(
                    "[bot] capture close error: "
                    f"{exc}"
                )

            # -------------------------------------------------
            # Đóng cửa sổ debug
            # -------------------------------------------------

            try:
                self._debug_window.close()

            except Exception as exc:
                print(
                    "[bot] debug window close error: "
                    f"{exc}"
                )

            # -------------------------------------------------
            # CHỈ bot sở hữu Monitor mới được đóng Monitor
            # -------------------------------------------------

            if self._owns_monitor:
                try:
                    self._monitor.close()

                except Exception as exc:
                    print(
                        "[bot] monitor close error: "
                        f"{exc}"
                    )