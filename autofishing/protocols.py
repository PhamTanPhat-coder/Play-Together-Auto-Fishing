"""Cross-cutting protocols."""

from __future__ import annotations

from typing import Protocol


class FrameSource(Protocol):
    def grab_bgr(self, *, full: bool = False): ...

    def close(self) -> None: ...


class Clicker(Protocol):
    def click(self, x: int, y: int) -> None: ...

    def press_key(self, key: str) -> None: ...
