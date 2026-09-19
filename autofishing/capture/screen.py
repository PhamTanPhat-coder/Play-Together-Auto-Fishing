from __future__ import annotations

import mss
import numpy as np

from autofishing.capture.region import Region


class ScreenCapture:
    def __init__(self, region: Region) -> None:
        self.region = region
        self._sct = mss.mss()

    def grab_bgr(self, *, full: bool = False) -> np.ndarray:
        del full  # absolute screen region — no crop toggle
        shot = self._sct.grab(self.region.as_mss_monitor())
        frame = np.array(shot, dtype=np.uint8)
        return frame[:, :, :3]

    def close(self) -> None:
        self._sct.close()
