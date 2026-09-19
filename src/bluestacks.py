"""Deprecated compatibility shim — use `autofishing.capture`."""

from autofishing.capture.bluestacks import BlueStacksCapture
from autofishing.capture.window import BlueStacksWindow, find_bluestacks_game_surface
from autofishing.input.clickers import BlueStacksClicker

__all__ = [
    "BlueStacksCapture",
    "BlueStacksClicker",
    "BlueStacksWindow",
    "find_bluestacks_game_surface",
]
