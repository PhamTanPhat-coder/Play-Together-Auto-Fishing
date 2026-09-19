from autofishing.bot.factory import build_bot_from_config
from autofishing.bot.fishing import BotState, FishingBot
from autofishing.bot.preview import preview_capture
from autofishing.config import BotSettings

# Alias used by older docs / shims
BotConfig = BotSettings

__all__ = [
    "BotConfig",
    "BotSettings",
    "BotState",
    "FishingBot",
    "build_bot_from_config",
    "preview_capture",
]
