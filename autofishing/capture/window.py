"""Locate BlueStacks game surface (Keymap Overlay / App Player)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pyautogui

try:
    import pygetwindow as gw
except ImportError:  # pragma: no cover
    gw = None


def _win32_focus(hwnd: int, settle_s: float = 0.05) -> None:
    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    user32.ShowWindow(hwnd, 9)  # SW_RESTORE

    fg = user32.GetForegroundWindow()
    cur_thread = kernel32.GetCurrentThreadId()
    fg_thread = user32.GetWindowThreadProcessId(fg, None)
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)

    user32.AttachThreadInput(cur_thread, fg_thread, True)
    user32.AttachThreadInput(fg_thread, target_thread, True)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(fg_thread, target_thread, False)
    user32.AttachThreadInput(cur_thread, fg_thread, False)
    if settle_s > 0:
        time.sleep(settle_s)


@dataclass
class BlueStacksWindow:
    title: str
    left: int
    top: int
    width: int
    height: int
    _hwnd: int | None = None

    def focus(self, settle_s: float = 0.05) -> None:
        if self._hwnd:
            try:
                _win32_focus(self._hwnd, settle_s=settle_s)
                return
            except Exception:
                pass
        pyautogui.click(self.left + 40, self.top + 10)
        time.sleep(settle_s)


def find_bluestacks_game_surface() -> BlueStacksWindow:
    if gw is None:
        raise RuntimeError("pygetwindow not installed")

    overlay = None
    app = None
    for w in gw.getAllWindows():
        if not w.title:
            continue
        if w.title == "BlueStacks Keymap Overlay" and w.width >= 800 and w.height >= 400:
            overlay = w
        if w.title == "BlueStacks App Player" and w.width >= 800 and w.height >= 400:
            app = w

    if overlay is not None:
        hwnd = getattr(app, "_hWnd", None) if app is not None else getattr(overlay, "_hWnd", None)
        return BlueStacksWindow(
            title=overlay.title,
            left=overlay.left,
            top=overlay.top,
            width=overlay.width,
            height=overlay.height,
            _hwnd=hwnd or getattr(overlay, "_hWnd", None),
        )

    if app is None:
        raise RuntimeError("BlueStacks App Player window not found.")

    inset_top = 32
    return BlueStacksWindow(
        title=app.title,
        left=app.left,
        top=app.top + inset_top,
        width=app.width,
        height=app.height - inset_top,
        _hwnd=getattr(app, "_hWnd", None),
    )
