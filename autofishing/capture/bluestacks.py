"""BlueStacks window capture via mss."""

from __future__ import annotations

import numpy as np

from autofishing.capture.window import find_bluestacks_game_surface


class BlueStacksCapture:
    """Capture BlueStacks overlay — much faster than ADB screencap."""

    def __init__(
        self,
        image_width: int = 1600,
        image_height: int = 900,
        roi_left: int = 0,
        roi_top: int = 0,
        roi_width: int | None = None,
        roi_height: int | None = None,
    ) -> None:
        import mss

        self.image_width = image_width
        self.image_height = image_height
        self.roi_left = roi_left
        self.roi_top = roi_top
        self.roi_width = roi_width
        self.roi_height = roi_height
        self._sct = mss.mss()
        self._surf = find_bluestacks_game_surface()
        print(
            f"[bs] capture overlay={self._surf.width}x{self._surf.height} "
            f"@ ({self._surf.left},{self._surf.top}) logical={image_width}x{image_height}"
        )

    def _monitor(self, *, full: bool = False) -> dict[str, int]:
        surf = self._surf
        sx = surf.width / self.image_width
        sy = surf.height / self.image_height
        if not full and self.roi_width and self.roi_height:
            return {
                "left": surf.left + int(self.roi_left * sx),
                "top": surf.top + int(self.roi_top * sy),
                "width": max(1, int(self.roi_width * sx)),
                "height": max(1, int(self.roi_height * sy)),
            }
        return {
            "left": surf.left,
            "top": surf.top,
            "width": surf.width,
            "height": surf.height,
        }

    def grab_bgr(self, *, full: bool = False) -> np.ndarray:
        shot = self._sct.grab(self._monitor(full=full))
        return np.ascontiguousarray(np.asarray(shot, dtype=np.uint8)[:, :, :3])

    def close(self) -> None:
        self._sct.close()
