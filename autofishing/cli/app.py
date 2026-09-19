"""CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from autofishing.bot.controller import BotController
from autofishing.bot.preview import preview_capture
from autofishing.cli import commands
from autofishing.config import load_yaml
from autofishing.monitor import FishingStatusMonitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play Together auto fishing bot"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview capture ROI",
    )

    parser.add_argument(
        "--test",
        type=Path,
        metavar="IMAGE",
        help="Detect on one image",
    )

    parser.add_argument(
        "--download-dataset",
        action="store_true",
    )

    parser.add_argument(
        "--download-model",
        action="store_true",
    )

    parser.add_argument(
        "--extract-zip",
        type=Path,
        metavar="ZIP",
    )

    parser.add_argument(
        "--status",
        action="store_true",
    )

    parser.add_argument(
        "--adb-test",
        action="store_true",
    )

    parser.add_argument(
        "--tap",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
    )

    parser.add_argument(
        "--space",
        action="store_true",
    )

    parser.add_argument(
        "--pick",
        action="store_true",
        help="Pick image coordinates",
    )

    parser.add_argument(
        "--click",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
    )

    parser.add_argument(
        "--api-key",
        help="Roboflow API key",
    )

    parser.add_argument(
        "--version",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/play-together2"),
    )

    return parser


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    api_key = commands.resolve_api_key(args.api_key)

    # =========================================================
    # ONE-SHOT COMMANDS
    # =========================================================

    if args.status:
        commands.print_status(cfg)
        return

    if args.click:
        commands.cmd_click(
            cfg,
            args.click[0],
            args.click[1],
        )
        return

    if args.pick:
        commands.cmd_pick(cfg)
        return

    if args.adb_test or args.tap or args.space:
        commands.cmd_adb_tools(
            cfg,
            tap=args.tap,
            space=args.space,
        )
        return

    if args.extract_zip:
        commands.extract_dataset_zip(
            args.extract_zip,
            args.output,
        )

        commands.print_status(cfg)
        return

    if args.download_dataset:
        if not api_key:
            parser.error(
                "Set ROBOFLOW_API_KEY "
                "in .env or pass --api-key"
            )

        commands.download_dataset(
            api_key,
            args.version,
            args.output,
        )

        commands.print_status(cfg)
        return

    if args.download_model:
        if not api_key:
            parser.error(
                "Set ROBOFLOW_API_KEY "
                "in .env or pass --api-key"
            )

        commands.download_model(
            api_key,
            args.version,
            Path(cfg["model"]["path"]),
        )

        commands.print_status(cfg)
        return

    if args.preview:
        preview_capture(cfg)
        return

    if args.test:
        commands.test_image(
            cfg,
            args.test,
        )
        return

    # =========================================================
    # GUI BOT CONTROLLER
    # =========================================================

    monitor = FishingStatusMonitor()

    controller = BotController(
        cfg=cfg,
        monitor=monitor,
    )

    # Monitor dùng controller để gọi:
    # START
    # PAUSE / RESUME
    # RESET
    # RESTART
    # STOP
    # AUTO RECOVERY
    # THOÁT
    monitor.set_controller(controller)

    # =========================================================
    # GIỮ HÀNH VI CŨ
    #
    # Chạy main.py -> bot tự khởi động.
    # Không cần bấm START lần đầu.
    # =========================================================

    controller.start()

    try:
        # RẤT QUAN TRỌNG:
        #
        # show() chạy Tkinter mainloop()
        # trực tiếp trên MAIN THREAD.
        #
        # KHÔNG đưa show() vào threading.Thread.
        monitor.show()

    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()