from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    top: int
    left: int
    width: int
    height: int

    def as_mss_monitor(self) -> dict[str, int]:
        return {
            "top": self.top,
            "left": self.left,
            "width": self.width,
            "height": self.height,
        }
