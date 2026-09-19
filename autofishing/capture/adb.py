"""ADB screencap / tap for BlueStacks."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AdbRegion:
    """Crop region in screencap (image) coordinates."""

    left: int = 0
    top: int = 0
    width: int | None = None
    height: int | None = None


class AdbDevice:
    """BlueStacks ADB: screencap (native orientation for YOLO) + tap mapping."""

    def __init__(
        self,
        serial: str = "127.0.0.1:5555",
        adb_path: str = "adb",
        region: AdbRegion | None = None,
    ) -> None:
        self.serial = serial
        self.adb_path = adb_path
        self.region = region or AdbRegion()
        self._last_shape: tuple[int, int] | None = None
        self._ensure_connected()

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        cmd = [self.adb_path, "-s", self.serial, *args]
        result = subprocess.run(cmd, capture_output=True)
        if check and result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"adb failed ({' '.join(args)}): {err or result.returncode}")
        return result

    def _ensure_connected(self) -> None:
        if ":" in self.serial and not self.serial.startswith("emulator-"):
            subprocess.run([self.adb_path, "connect", self.serial], capture_output=True)
        devices = subprocess.run(
            [self.adb_path, "devices"], capture_output=True, text=True
        ).stdout
        lines = [ln for ln in devices.splitlines() if self.serial in ln]
        if not lines or "device" not in lines[0]:
            raise RuntimeError(
                f"ADB device not ready: {self.serial}\n{devices}\n"
                f"Enable ADB then: adb connect {self.serial}"
            )

    def screen_size(self) -> tuple[int, int]:
        out = self._run("shell", "wm", "size").stdout.decode("utf-8", errors="replace")
        for part in out.replace("\r", "").split():
            if "x" in part and part[0].isdigit():
                w, h = part.split("x")
                return int(w), int(h)
        raise RuntimeError(f"Cannot parse wm size: {out}")

    def grab_bgr(self, *, full: bool = False) -> np.ndarray:
        result = self._run("exec-out", "screencap")
        raw = result.stdout
        if len(raw) < 16:
            raise RuntimeError("ADB screencap empty")

        w = int.from_bytes(raw[0:4], "little")
        h = int.from_bytes(raw[4:8], "little")
        payload = raw[12:]
        expected = w * h * 4
        if w <= 0 or h <= 0 or len(payload) < expected:
            png = self._run("exec-out", "screencap", "-p").stdout
            img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to decode ADB screencap")
        else:
            rgba = np.frombuffer(payload, dtype=np.uint8, count=expected).reshape((h, w, 4))
            img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        self._last_shape = img.shape[:2]
        r = self.region
        if not full and r.width and r.height:
            img = img[r.top : r.top + r.height, r.left : r.left + r.width]
        return img

    def image_to_tap(self, x: int, y: int) -> tuple[int, int]:
        tap_w, tap_h = self.screen_size()
        if self._last_shape is None:
            self.grab_bgr()
        assert self._last_shape is not None
        img_h, img_w = self._last_shape

        fx = x + (self.region.left if self.region.width else 0)
        fy = y + (self.region.top if self.region.height else 0)

        if (img_w, img_h) == (tap_w, tap_h):
            return int(fx), int(fy)
        if (img_w, img_h) == (tap_h, tap_w):
            return int(img_h - 1 - fy), int(fx)
        return int(fx * tap_w / img_w), int(fy * tap_h / img_h)

    def tap(self, x: int, y: int, *, image_coords: bool = True) -> None:
        if image_coords:
            tx, ty = self.image_to_tap(x, y)
        else:
            tx, ty = int(x), int(y)
        self._run("shell", "input", "tap", str(tx), str(ty))
        print(f"[adb] tap image=({int(x)},{int(y)}) -> wm=({tx},{ty})")

    def keyevent(self, keycode: int) -> None:
        self._run("shell", "input", "keyevent", str(int(keycode)))
        print(f"[adb] keyevent {keycode}")

    def press_space(self) -> None:
        self.keyevent(62)

    def tap_center(self) -> None:
        if self._last_shape is None:
            self.grab_bgr()
        assert self._last_shape is not None
        h, w = self._last_shape
        self.tap(w // 2, h // 2)

    def save_screenshot(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.grab_bgr())
        return path

    def close(self) -> None:
        pass
