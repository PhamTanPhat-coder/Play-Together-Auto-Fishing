"""YAML config loading and typed bot settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class BotSettings:
    poll_interval_ms: int
    click_delay_ms: int
    cooldown_after_bite_ms: int
    min_wait_before_bite_ms: int
    cast_x: int
    cast_y: int
    reel_x: int
    reel_y: int
    store_x: int = 1153
    store_y: int = 769
    bag_x: int | None = None
    bag_y: int | None = None
    auto_cast: bool = True
    click_on_detection: bool = False
    cast_delay_ms: int = 800
    after_catch_min_ms: int = 300
    after_catch_timeout_ms: int = 12000
    repair_timeout_ms: int = 20000
    repair_click_gap_ms: int = 700
    allow_paid_repair: bool = True
    ui_imgsz: int = 416
    ui_confidence: float = 0.35
    action_mode: str = "tap"
    action_key: str = "space"
    cast_mode: str = "tap"
    jerk_mode: str = "tap"
    store_mode: str = "tap"


def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def bot_settings_from_dict(cfg: dict) -> BotSettings:
    model_cfg = cfg.get("model", {})
    bot_cfg = cfg["bot"]
    cast_x = int(bot_cfg.get("cast_x", bot_cfg.get("click_x", 780)))
    cast_y = int(bot_cfg.get("cast_y", bot_cfg.get("click_y", 1000)))
    reel_x = int(bot_cfg.get("reel_x", cast_x))
    reel_y = int(bot_cfg.get("reel_y", cast_y))
    bag_x = bot_cfg.get("bag_x")
    bag_y = bot_cfg.get("bag_y")
    default_mode = bot_cfg.get("action_mode", "tap")

    return BotSettings(
        poll_interval_ms=bot_cfg["poll_interval_ms"],
        click_delay_ms=bot_cfg["click_delay_ms"],
        cooldown_after_bite_ms=bot_cfg["cooldown_after_bite_ms"],
        min_wait_before_bite_ms=bot_cfg["min_wait_before_bite_ms"],
        cast_x=cast_x,
        cast_y=cast_y,
        reel_x=reel_x,
        reel_y=reel_y,
        store_x=int(bot_cfg.get("store_x", 1153)),
        store_y=int(bot_cfg.get("store_y", 769)),
        bag_x=int(bag_x) if bag_x is not None else None,
        bag_y=int(bag_y) if bag_y is not None else None,
        auto_cast=bot_cfg.get("auto_cast", True),
        click_on_detection=bot_cfg.get("click_on_detection", False),
        cast_delay_ms=bot_cfg.get("cast_delay_ms", 800),
        after_catch_min_ms=bot_cfg.get("after_catch_min_ms", 300),
        after_catch_timeout_ms=bot_cfg.get("after_catch_timeout_ms", 12000),
        repair_timeout_ms=bot_cfg.get("repair_timeout_ms", 20000),
        repair_click_gap_ms=bot_cfg.get("repair_click_gap_ms", 700),
        allow_paid_repair=bool(bot_cfg.get("allow_paid_repair", True)),
        ui_imgsz=int(bot_cfg.get("ui_imgsz", model_cfg.get("ui_imgsz", 416))),
        ui_confidence=float(
            bot_cfg.get("ui_confidence", model_cfg.get("ui_confidence", 0.35))
        ),
        action_mode=default_mode,
        action_key=bot_cfg.get("action_key", "space"),
        cast_mode=bot_cfg.get("cast_mode", default_mode),
        jerk_mode=bot_cfg.get("jerk_mode", default_mode),
        store_mode=bot_cfg.get("store_mode", default_mode),
    )


def resolve_adb_path(cfg: dict) -> str:
    return cfg.get("adb", {}).get("path") or "adb"
