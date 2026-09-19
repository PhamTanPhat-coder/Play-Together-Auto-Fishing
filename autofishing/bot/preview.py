"""Live capture preview helper."""

from __future__ import annotations

import time

import cv2

from autofishing.capture.adb import AdbDevice, AdbRegion
from autofishing.capture.bluestacks import BlueStacksCapture
from autofishing.capture.region import Region
from autofishing.capture.screen import ScreenCapture
from autofishing.config import resolve_adb_path
from autofishing.protocols import FrameSource


def preview_capture(cfg: dict) -> None:
    backend = cfg.get("backend", "screen")
    cap_cfg = cfg.get("capture", {})
    img_w = int(cap_cfg.get("image_width", 1600))
    img_h = int(cap_cfg.get("image_height", 900))
    roi_w = cap_cfg.get("width")
    roi_h = cap_cfg.get("height")
    roi_left = int(cap_cfg.get("left", 0))
    roi_top = int(cap_cfg.get("top", 0))

    if backend in ("window", "bluestacks", "bs"):
        capture: FrameSource = BlueStacksCapture(
            image_width=img_w,
            image_height=img_h,
            roi_left=roi_left,
            roi_top=roi_top,
            roi_width=int(roi_w) if roi_w else None,
            roi_height=int(roi_h) if roi_h else None,
        )
    elif backend == "adb":
        region = AdbRegion(
            left=roi_left,
            top=roi_top,
            width=int(roi_w) if roi_w else None,
            height=int(roi_h) if roi_h else None,
        )
        if not region.width or not region.height:
            region = AdbRegion()
        capture = AdbDevice(
            serial=cfg.get("adb", {}).get("serial", "127.0.0.1:5555"),
            adb_path=resolve_adb_path(cfg),
            region=region,
        )
    else:
        capture = ScreenCapture(
            Region(
                top=int(cap_cfg["top"]),
                left=int(cap_cfg["left"]),
                width=int(cap_cfg["width"]),
                height=int(cap_cfg["height"]),
            )
        )

    bot_cfg = cfg.get("bot", {})
    cast_x = int(bot_cfg.get("cast_x", 0))
    cast_y = int(bot_cfg.get("cast_y", 0))
    draw_x = cast_x - roi_left if roi_w else cast_x
    draw_y = cast_y - roi_top if roi_h else cast_y

    print("[preview] press Q to quit — green circle = cast (if inside ROI)")
    try:
        while True:
            t0 = time.perf_counter()
            frame = capture.grab_bgr()
            grab_ms = (time.perf_counter() - t0) * 1000
            if (
                cast_x
                and cast_y
                and 0 <= draw_x < frame.shape[1]
                and 0 <= draw_y < frame.shape[0]
            ):
                cv2.circle(frame, (draw_x, draw_y), 24, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"grab {grab_ms:.0f}ms",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("capture preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.close()
        cv2.destroyAllWindows()
