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
from autofishing.input.clickers import BlueStacksClicker, ScreenClicker
from autofishing.monitor import FishingStatusMonitor, StatusSnapshot
from autofishing.protocols import Clicker, FrameSource


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

    def start(self) -> None:
        """Khởi động cửa sổ debug."""
        if self._started_once:
            return

        self._started_once = True

        self._thread.start()

        self._started.wait(timeout=3.0)

    def update(
        self,
        frame,
        bite: Detection | None,
    ) -> None:
        """Nhận frame mới và tạo PNG tạm."""

        if self._closed_by_user:
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

        # Viền ROI.
        cv2.rectangle(
            debug_frame,
            (0, 0),
            (w - 1, h - 1),
            (0, 255, 0),
            2,
        )

        # Tâm ROI.
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

        # Kích thước ROI.
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

        # Thông tin detection.
        if bite is not None:
            cv2.rectangle(
                debug_frame,
                (bite.x1, bite.y1),
                (bite.x2, bite.y2),
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

        # Resize về kích thước cửa sổ.
        display = cv2.resize(
            debug_frame,
            (
                self.WINDOW_WIDTH,
                self.WINDOW_HEIGHT,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # Encode PNG.
        ok, encoded = cv2.imencode(
            ".png",
            display,
        )

        if not ok:
            return

        png_bytes = encoded.tobytes()

        # Tạo file PNG tạm.
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
                file.write(png_bytes)

        except Exception as exc:
            if not self._image_error_reported:
                print(
                    "[debug-window] "
                    f"PNG write error: {exc}"
                )

                self._image_error_reported = True

            return

        # Xóa frame cũ khỏi queue.
        try:
            while True:
                old_path = (
                    self._queue.get_nowait()
                )

                try:
                    os.remove(old_path)
                except OSError:
                    pass

        except queue.Empty:
            pass

        # Đưa frame mới nhất vào queue.
        try:
            self._queue.put_nowait(
                file_path
            )

        except queue.Full:
            try:
                os.remove(file_path)
            except OSError:
                pass

    def close(self) -> None:
        """Đóng cửa sổ debug."""

        self._stop_event.set()

        root = self._root

        if root is not None:
            try:
                root.after(
                    0,
                    root.destroy,
                )
            except Exception:
                pass

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

    def _pump_frame(self) -> None:
        """Đọc PNG mới nhất và hiển thị."""

        root = self._root

        if root is None:
            return

        if self._stop_event.is_set():
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

    def _keep_topmost(self) -> None:
        """Giữ cửa sổ luôn nằm trên BlueStacks."""

        root = self._root

        if root is None:
            return

        if self._stop_event.is_set():
            return

        self._make_topmost()
        self._show_without_activate()

        root.after(
            200,
            self._keep_topmost,
        )

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

    def _on_user_close(self) -> None:
        self._closed_by_user = True

        self._stop_event.set()

        root = self._root

        if root is not None:
            root.destroy()

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
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.config = config

        self.clicker = (
            clicker
            or ScreenClicker()
        )

        self.state = (
            BotState.CAST
            if config.auto_cast
            else BotState.WAITING_BITE
        )

        self._state_entered_at = (
            time.monotonic()
        )

        self._running = False

        self._last_repair_click_at = 0.0
        self._repair_opened_bag = False
        self._too_late_seen_at: float | None = None

        # Monitoring.
        self._monitor = (
            FishingStatusMonitor()
        )

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

        # -----------------------------------------------------
        # Đảm bảo một lượt câu chỉ ghi nhận hụt 1 lần.
        # -----------------------------------------------------
        self._miss_recorded_for_cast = False

        # Debug window.
        self._debug_window_enabled = True

        self._debug_window = (
            _BiteDebugWindow()
        )

    # =========================================================
    # STATE LABEL
    # =========================================================

    def _state_label(self) -> str:
        return _STATE_LABELS.get(
            self.state,
            self.state.value,
        )

    # =========================================================
    # STATUS
    # =========================================================

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

    # =========================================================
    # RECORD MISS
    # =========================================================

    def _record_miss(
        self,
        reason: str,
    ) -> bool:
        """
        Ghi nhận 1 lượt câu hụt.

        Trả về:
            True  = vừa ghi nhận hụt.
            False = lượt này đã được ghi hụt trước đó.

        Nhờ đó một lượt câu không thể bị cộng hụt 2 lần.
        """

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
                x + self.capture.region.left,
                y + self.capture.region.top,
            )

        if (
            from_roi
            and isinstance(
                self.capture,
                ScreenCapture,
            )
        ):
            return (
                x + self.capture.region.left,
                y + self.capture.region.top,
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

        # -----------------------------------------------------
        # Bắt đầu một lượt câu mới.
        # Reset cờ hụt của lượt trước.
        # -----------------------------------------------------

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
        # 45 giây timeout.
        if self._elapsed_ms() >= 45000:
            self._event(
                "45s chưa thấy ! → reset vòng câu"
            )

            self._set_state(
                BotState.COOLDOWN
            )

            return

        # Không detect trước thời gian quy định.
        if (
            self._elapsed_ms()
            < self.config.min_wait_before_bite_ms
        ):
            return

        # -----------------------------------------------------
        # Scan UI mỗi 1.5 giây.
        # -----------------------------------------------------

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

            # -------------------------------------------------
            # Repair
            # -------------------------------------------------

            if self._ui_requests_repair(
                ui
            ):
                self._event(
                    "Phát hiện giao diện sửa "
                    "khi đang chờ cá"
                )

                self._set_state(
                    BotState.REPAIR
                )

                return

            # -------------------------------------------------
            # Too late trước reel
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Store còn sót lại
            # -------------------------------------------------

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
        # Capture ROI.
        # -----------------------------------------------------

        t0 = time.perf_counter()

        frame = self._grab(
            full=False
        )

        t1 = time.perf_counter()

        # -----------------------------------------------------
        # YOLO.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Debug window.
        # -----------------------------------------------------

        self._show_bite_debug(
            frame,
            bite,
        )

        # -----------------------------------------------------
        # Không thấy !
        # -----------------------------------------------------

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
        # Phát hiện !
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

        # -----------------------------------------------------
        # Reel.
        # -----------------------------------------------------

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
        # Bắt được cá
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
        # Repair
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
        # Too late sau khi reel
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
        # Timeout popup.
        #
        # Đây là trường hợp:
        # - đã reel
        # - không thấy Store
        # - không thấy too-late
        #
        # Ta tính 1 lượt hụt/không xác nhận.
        # Sau đó vẫn Store fallback.
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
                f"Store fallback "
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

        # -----------------------------------------------------
        # Heartbeat.
        # -----------------------------------------------------

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
                "Sửa cần timeout → cooldown"
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

        # Repair done.
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

        # Paid repair.
        if (
            "paid-repair-button"
            in ui
            and self._can_repair_click()
        ):
            if not self.config.allow_paid_repair:
                self._event(
                    "Paid repair bị chặn"
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

        # Repair button.
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

        # Tool tab.
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

        # Open bag.
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
                    "Chưa set bag_x/bag_y "
                    "trong config.yaml "
                    "(python main.py --pick "
                    "trên icon balo)"
                )

            return

        # Heartbeat repair.
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

        self._monitor.start()

        self._publish_status()

        # Mở cửa sổ debug.
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
            self._running = False

            self._monitor.close()

            self.capture.close()

            self._debug_window.close()

    # =========================================================
    # STOP
    # =========================================================

    def stop(self) -> None:
        self._running = False