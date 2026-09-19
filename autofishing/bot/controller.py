"""Điều khiển vòng đời bot câu cá."""

from __future__ import annotations

import threading
import time
from typing import Any

from autofishing.bot.factory import build_bot_from_config


class BotController:
    """Quản lý START / PAUSE / RESET / RESTART / STOP."""

    def __init__(
        self,
        cfg: dict,
        monitor,
    ) -> None:
        self.cfg = cfg
        self.monitor = monitor

        self._bot = None
        self._thread: threading.Thread | None = None

        self._lock = threading.RLock()

        self._stop_requested = False
        self._restart_requested = False

        self._auto_recovery = True

        self._saved_statistics: dict[str, Any] = {
            "casts": 0,
            "caught": 0,
            "missed": 0,
            "no_detect": 0,
            "bites": 0,
            "fallback_store": 0,
            "last_conf": None,
            "last_event": "-",
            "started_at": 0.0,
        }

    # =========================================================
    # START
    # =========================================================

    def start(self) -> None:
        """Bắt đầu bot hoặc tiếp tục bot đang tạm dừng."""

        with self._lock:
            # Bot đang chạy.
            if (
                self._thread is not None
                and self._thread.is_alive()
            ):
                if self._bot is not None:
                    try:
                        self._bot.resume()
                    except Exception:
                        pass

                self._log(
                    "Đã tiếp tục bot."
                )

                return

            self._stop_requested = False
            self._restart_requested = False

            self._thread = threading.Thread(
                target=self._run_worker,
                name="FishingBotWorker",
                daemon=True,
            )

            self._thread.start()

        self._log(
            "Đang khởi động bot..."
        )

    # =========================================================
    # VÒNG CHẠY CHÍNH
    # =========================================================

    def _run_worker(self) -> None:
        while True:
            with self._lock:
                if self._stop_requested:
                    self._bot = None
                    return

                restart = self._restart_requested

                if restart:
                    self._restart_requested = False

            # -------------------------------------------------
            # Tạo BOT MỚI
            # -------------------------------------------------

            bot = None

            try:
                bot = build_bot_from_config(
                    self.cfg,
                    monitor=self.monitor,
                )

                # Khôi phục số liệu cũ khi restart.
                self._restore_statistics(
                    bot
                )

                with self._lock:
                    self._bot = bot

                self._log(
                    "Bot đã khởi động."
                )

                # -------------------------------------------------
                # CHẠY BOT
                # -------------------------------------------------

                bot.run()

                # -------------------------------------------------
                # BOT DỪNG BÌNH THƯỜNG
                # -------------------------------------------------

                self._save_statistics(
                    bot
                )

            except Exception as exc:
                if bot is not None:
                    self._save_statistics(
                        bot
                    )

                self._log(
                    f"Bot gặp lỗi: {exc}"
                )

                with self._lock:
                    should_stop = (
                        self._stop_requested
                    )

                    auto_recovery = (
                        self._auto_recovery
                    )

                if should_stop:
                    break

                if not auto_recovery:
                    self._log(
                        "Bot đã dừng vì "
                        "tự động phục hồi đang tắt."
                    )
                    break

                self._log(
                    "Bot sẽ tự chạy lại sau 2 giây."
                )

                time.sleep(2)

                continue

            finally:
                with self._lock:
                    self._bot = None

            # -------------------------------------------------
            # SAU KHI BOT RUN KẾT THÚC
            # -------------------------------------------------

            with self._lock:
                should_stop = (
                    self._stop_requested
                )

                should_restart = (
                    self._restart_requested
                )

                auto_recovery = (
                    self._auto_recovery
                )

            if should_stop:
                break

            if should_restart:
                self._log(
                    "Bot cũ đã dừng."
                )

                self._log(
                    "Đang khởi động bot mới..."
                )

                continue

            # Bot tự nhiên kết thúc mà không phải
            # do nút Dừng hoặc Khởi động lại.
            if auto_recovery:
                self._log(
                    "Bot đã dừng bất thường."
                )

                self._log(
                    "Đang tự khởi động lại..."
                )

                time.sleep(2)

                continue

            break

        self._log(
            "Bot đã dừng."
        )

    # =========================================================
    # TẠM DỪNG
    # =========================================================

    def toggle_pause(self) -> None:
        with self._lock:
            bot = self._bot

        if bot is None:
            self._log(
                "Bot chưa chạy."
            )
            return

        try:
            if bot.is_paused:
                bot.resume()

                self._log(
                    "Đã tiếp tục bot."
                )

            else:
                bot.pause()

                self._log(
                    "Đã tạm dừng bot."
                )

        except Exception as exc:
            self._log(
                f"Không thể tạm dừng/tiếp tục: {exc}"
            )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:
        with self._lock:
            bot = self._bot

            self._saved_statistics = {
                "casts": 0,
                "caught": 0,
                "missed": 0,
                "no_detect": 0,
                "bites": 0,
                "fallback_store": 0,
                "last_conf": None,
                "last_event": "-",
                "started_at": (
                    time.monotonic()
                ),
            }

        if bot is not None:
            try:
                bot.reset_stats()
            except Exception:
                pass

        self._log(
            "Đã đặt lại số liệu."
        )

    # =========================================================
    # KHỞI ĐỘNG LẠI
    # =========================================================

    def restart(self) -> None:
        """
        Khởi động lại thật sự:

        1. Dừng bot hiện tại.
        2. Cho bot cũ thoát hoàn toàn.
        3. Đóng capture theo vòng đời của FishingBot.
        4. Tạo bot mới.
        5. Tự chạy bot mới.
        6. Giữ lại số liệu cũ.
        """

        with self._lock:
            self._restart_requested = True
            self._stop_requested = False

            bot = self._bot

            thread = self._thread

        self._log(
            "Đang khởi động lại bot..."
        )

        if bot is not None:
            try:
                bot.stop()
            except Exception as exc:
                self._log(
                    f"Lỗi khi dừng bot cũ: {exc}"
                )

        # Không join trực tiếp ở GUI thread.
        # Worker sẽ tự thoát rồi tạo bot mới.
        if (
            thread is None
            or not thread.is_alive()
        ):
            with self._lock:
                self._stop_requested = False

                self._thread = threading.Thread(
                    target=self._run_worker,
                    name="FishingBotWorker",
                    daemon=True,
                )

                self._thread.start()

    # =========================================================
    # STOP
    # =========================================================

    def stop(
        self,
        wait: bool = False,
    ) -> None:
        with self._lock:
            self._stop_requested = True
            self._restart_requested = False

            bot = self._bot
            thread = self._thread

        if bot is not None:
            try:
                bot.stop()
            except Exception as exc:
                self._log(
                    f"Lỗi khi dừng bot: {exc}"
                )

        self._log(
            "Đã yêu cầu dừng bot."
        )

        if (
            wait
            and thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(
                timeout=10
            )

    # =========================================================
    # TỰ PHỤC HỒI
    # =========================================================

    def set_auto_recovery(
        self,
        enabled: bool,
    ) -> None:
        with self._lock:
            self._auto_recovery = bool(
                enabled
            )

        if enabled:
            self._log(
                "Đã bật tự động chạy lại khi bot gặp lỗi."
            )
        else:
            self._log(
                "Đã tắt tự động chạy lại khi bot gặp lỗi."
            )

    # =========================================================
    # SHUTDOWN
    # =========================================================

    def shutdown(self) -> None:
        with self._lock:
            self._stop_requested = True
            self._restart_requested = False

            bot = self._bot
            thread = self._thread

        if bot is not None:
            try:
                bot.stop()
            except Exception:
                pass

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(
                timeout=10
            )

    # =========================================================
    # THỐNG KÊ
    # =========================================================

    def _save_statistics(
        self,
        bot,
    ) -> None:
        try:
            statistics = bot.export_statistics()

        except Exception:
            statistics = None

        if isinstance(
            statistics,
            dict,
        ):
            with self._lock:
                self._saved_statistics = dict(
                    statistics
                )

    def _restore_statistics(
        self,
        bot,
    ) -> None:
        try:
            bot.restore_statistics(
                dict(
                    self._saved_statistics
                )
            )

        except Exception:
            pass

    # =========================================================
    # LOG
    # =========================================================

    def _log(
        self,
        message: str,
    ) -> None:
        try:
            print(
                f"[bot] {message}"
            )
        except Exception:
            pass

        try:
            self.monitor.log_event(
                message
            )
        except Exception:
            pass