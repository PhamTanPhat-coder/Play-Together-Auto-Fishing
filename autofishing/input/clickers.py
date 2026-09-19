"""Mouse / keyboard backends."""

from __future__ import annotations

import pyautogui

from autofishing.capture.adb import AdbDevice
from autofishing.capture.window import BlueStacksWindow, find_bluestacks_game_surface

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class ScreenClicker:
    def click(self, x: int, y: int) -> None:
        pyautogui.click(x, y)
        print(f"[bot] screen click ({x}, {y})")

    def press_key(self, key: str) -> None:
        self._focus_bluestacks()
        pyautogui.press(key)
        print(f"[bot] key '{key}' (Windows → BlueStacks)")

    @staticmethod
    def _focus_bluestacks() -> None:
        try:
            BlueStacksClicker().focus()
        except Exception:
            pass


class AdbClicker:
    """ADB tap — often ignored by BlueStacks games; kept as fallback."""

    def __init__(self, device: AdbDevice) -> None:
        self.device = device

    def click(self, x: int, y: int) -> None:
        self.device.tap(x, y)

    def press_key(self, key: str) -> None:
        ScreenClicker().press_key(key)


class BlueStacksClicker:
    """Focus BlueStacks once, then click using logical image coordinates."""

    def __init__(self, image_width: int = 1600, image_height: int = 900) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self._surf: BlueStacksWindow | None = None
        self._focused = False

    def _surface(self) -> BlueStacksWindow:
        if self._surf is None:
            self._surf = find_bluestacks_game_surface()
        return self._surf

    def focus(self, force: bool = False, settle_s: float = 0.03) -> None:
        if self._focused and not force:
            return
        self._surface().focus(settle_s=settle_s)
        self._focused = True

    def click(self, x: int, y: int, *, refocus: bool = False) -> None:
        need_focus = refocus or not self._focused
        self.focus(force=refocus, settle_s=0.03 if need_focus else 0)
        surf = self._surface()
        sx = surf.left + int(x * surf.width / self.image_width)
        sy = surf.top + int(y * surf.height / self.image_height)
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetCursorPos(int(sx), int(sy))
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)
        except Exception:
            pyautogui.click(sx, sy)
        print(f"[bs] click image=({x},{y}) -> screen=({sx},{sy})")

    def press_key(self, key: str) -> None:
        self.focus()
        pyautogui.press(key)
        print(f"[bs] key '{key}'")
