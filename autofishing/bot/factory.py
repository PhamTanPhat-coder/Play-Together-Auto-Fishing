"""Assemble capture + detector + clicker from config.yaml."""

from __future__ import annotations

from autofishing.capture.adb import (
    AdbDevice,
    AdbRegion,
)
from autofishing.capture.bluestacks import (
    BlueStacksCapture,
)
from autofishing.capture.region import Region
from autofishing.capture.screen import ScreenCapture
from autofishing.config import (
    bot_settings_from_dict,
    resolve_adb_path,
)
from autofishing.detection.detector import YoloDetector
from autofishing.input.clickers import (
    BlueStacksClicker,
    ScreenClicker,
)
from autofishing.bot.fishing import FishingBot
from autofishing.monitor import FishingStatusMonitor


def build_bot_from_config(
    cfg: dict,
    monitor: FishingStatusMonitor | None = None,
) -> FishingBot:
    model_cfg = cfg["model"]

    backend = cfg.get(
        "backend",
        "screen",
    )

    settings = bot_settings_from_dict(
        cfg
    )

    detector = YoloDetector(
        model_path=model_cfg["path"],
        confidence=model_cfg["confidence"],
        bite_classes=model_cfg.get(
            "bite_classes"
        ),
        imgsz=int(
            model_cfg.get(
                "imgsz",
                320,
            )
        ),
    )

    cap_cfg = cfg.get(
        "capture",
        {},
    )

    img_w = int(
        cap_cfg.get(
            "image_width",
            1600,
        )
    )

    img_h = int(
        cap_cfg.get(
            "image_height",
            900,
        )
    )

    roi_w = cap_cfg.get(
        "width"
    )

    roi_h = cap_cfg.get(
        "height"
    )

    roi_left = int(
        cap_cfg.get(
            "left",
            0,
        )
    )

    roi_top = int(
        cap_cfg.get(
            "top",
            0,
        )
    )

    # =========================================================
    # WINDOW / BLUESTACKS
    # =========================================================

    if backend in (
        "window",
        "bluestacks",
        "bs",
    ):
        capture = BlueStacksCapture(
            image_width=img_w,
            image_height=img_h,
            roi_left=roi_left,
            roi_top=roi_top,
            roi_width=(
                int(roi_w)
                if roi_w
                else None
            ),
            roi_height=(
                int(roi_h)
                if roi_h
                else None
            ),
        )

        clicker = BlueStacksClicker(
            image_width=img_w,
            image_height=img_h,
        )

        print(
            f"[bot] backend=window "
            f"logical={img_w}x{img_h}"
        )

        return FishingBot(
            capture,
            detector,
            settings,
            clicker=clicker,
            monitor=monitor,
        )

    # =========================================================
    # ADB
    # =========================================================

    if backend == "adb":
        region = AdbRegion(
            left=roi_left,
            top=roi_top,
            width=(
                int(roi_w)
                if roi_w
                else None
            ),
            height=(
                int(roi_h)
                if roi_h
                else None
            ),
        )

        if (
            not region.width
            or not region.height
        ):
            region = AdbRegion()

            print(
                "[adb] detect ROI: full screen"
            )

        else:
            print(
                f"[adb] detect ROI: "
                f"left={region.left} "
                f"top={region.top} "
                f"{region.width}x"
                f"{region.height}"
            )

        device = AdbDevice(
            serial=cfg.get(
                "adb",
                {},
            ).get(
                "serial",
                "127.0.0.1:5555",
            ),
            adb_path=resolve_adb_path(
                cfg
            ),
            region=region,
        )

        w, h = device.screen_size()

        print(
            f"[adb] connected "
            f"{device.serial} "
            f"wm={w}x{h}"
        )

        clicker = BlueStacksClicker(
            image_width=img_w,
            image_height=img_h,
        )

        return FishingBot(
            device,
            detector,
            settings,
            clicker=clicker,
            monitor=monitor,
        )

    # =========================================================
    # SCREEN
    # =========================================================

    capture = ScreenCapture(
        Region(
            top=int(
                cap_cfg["top"]
            ),
            left=int(
                cap_cfg["left"]
            ),
            width=int(
                cap_cfg["width"]
            ),
            height=int(
                cap_cfg["height"]
            ),
        )
    )

    return FishingBot(
        capture,
        detector,
        settings,
        clicker=ScreenClicker(),
        monitor=monitor,
    )