"""CLI subcommands: status, pick, click, ADB, Roboflow helpers."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import yaml


def resolve_api_key(cli_key: str | None) -> str | None:
    return cli_key or os.getenv("ROBOFLOW_API_KEY")


def extract_dataset_zip(zip_path: Path, output: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output)
    data_yaml = output / "data.yaml"
    if data_yaml.exists():
        text = data_yaml.read_text(encoding="utf-8")
        text = (
            text.replace("../train/", "train/")
            .replace("../valid/", "valid/")
            .replace("../test/", "test/")
        )
        data_yaml.write_text(text, encoding="utf-8")
    print(f"[extract] dataset unpacked to {output}")


def print_status(cfg: dict) -> None:
    model_path = Path(cfg["model"]["path"])
    dataset_yaml = Path("datasets/data.yaml")
    has_api_key = bool(resolve_api_key(None))
    print("=== AutoFishing status ===")
    print(f"model:   {'OK' if model_path.exists() else 'MISSING'} ({model_path})")
    print(f"dataset: {'OK' if dataset_yaml.exists() else 'MISSING'} ({dataset_yaml})")
    print(f"api key: {'OK' if has_api_key else 'MISSING'} (.env or --api-key)")
    if dataset_yaml.exists():
        data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        print(f"classes: {data.get('names')}")


def test_image(cfg: dict, image_path: Path) -> None:
    import cv2

    from autofishing.detection.detector import YoloDetector

    model_cfg = cfg["model"]
    detector = YoloDetector(
        model_path=model_cfg["path"],
        confidence=model_cfg["confidence"],
        bite_classes=model_cfg.get("bite_classes"),
        imgsz=int(model_cfg.get("imgsz", 320)),
    )
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detections = detector.detect(frame)
    if not detections:
        print("[test] no detections")
        return

    for det in detections:
        print(
            f"[test] {det.class_name} conf={det.confidence:.2f} "
            f"box=({det.x1},{det.y1})-({det.x2},{det.y2})"
        )
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{det.class_name} {det.confidence:.2f}",
            (det.x1, max(det.y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imshow("test detection", frame)
    print("[test] press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def download_dataset(api_key: str, version: int, output: Path) -> None:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("rick-hsu-nv7rx").project("play-together2")
    project.version(version).download("yolov8", location=str(output))
    print(f"[download] dataset saved to {output}")


def download_model(api_key: str, version: int, output: Path) -> None:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("rick-hsu-nv7rx").project("play-together2")
    models = project.version(version).models
    if not models:
        raise RuntimeError(f"No trained models found for version {version}")
    output.parent.mkdir(parents=True, exist_ok=True)
    models[0].download(str(output.parent))
    downloaded = next(output.parent.glob("*.pt"), None)
    if downloaded and downloaded != output:
        downloaded.replace(output)
    print(f"[download] model saved to {output}")


def cmd_click(cfg: dict, x: int, y: int) -> None:
    from autofishing.capture.adb import AdbDevice
    from autofishing.input.clickers import BlueStacksClicker

    adb_cfg = cfg.get("adb", {})
    device = AdbDevice(
        serial=adb_cfg.get("serial", "127.0.0.1:5555"),
        adb_path=adb_cfg.get("path") or "adb",
    )
    frame = device.grab_bgr()
    fh, fw = frame.shape[:2]
    clicker = BlueStacksClicker(image_width=fw, image_height=fh)
    clicker.click(x, y)
    print("[test] done — check BlueStacks")


def cmd_pick(cfg: dict) -> None:
    import cv2

    from autofishing.capture.adb import AdbDevice

    adb_cfg = cfg.get("adb", {})
    device = AdbDevice(
        serial=adb_cfg.get("serial", "127.0.0.1:5555"),
        adb_path=adb_cfg.get("path") or "adb",
    )
    frame = device.grab_bgr()
    clone = frame.copy()
    print(f"[pick] frame={frame.shape[1]}x{frame.shape[0]} — click, Q to quit")
    print("[pick] each click prints coords for config.yaml")

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        view = clone.copy()
        cv2.circle(view, (x, y), 12, (0, 255, 0), 2)
        cv2.putText(
            view,
            f"({x}, {y})",
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("pick coords", view)
        print(f"cast_x: {x}")
        print(f"cast_y: {y}")
        print(f"reel_x: {x}")
        print(f"reel_y: {y}")
        print("---")

    cv2.namedWindow("pick coords", cv2.WINDOW_NORMAL)
    cv2.imshow("pick coords", frame)
    cv2.setMouseCallback("pick coords", on_mouse)
    while True:
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def cmd_adb_tools(
    cfg: dict, *, tap: list[int] | None = None, space: bool = False
) -> None:
    from autofishing.capture.adb import AdbDevice
    from autofishing.input.clickers import ScreenClicker

    adb_cfg = cfg.get("adb", {})
    device = AdbDevice(
        serial=adb_cfg.get("serial", "127.0.0.1:5555"),
        adb_path=adb_cfg.get("path") or "adb",
    )
    w, h = device.screen_size()
    print(f"[adb] {device.serial} size={w}x{h}")
    if space:
        ScreenClicker().press_key("space")
        print("[test] pressed Space")
        return
    shot = device.save_screenshot("datasets/adb-preview.png")
    print(f"[adb] screenshot -> {shot}")
    if tap:
        device.tap(tap[0], tap[1])
    else:
        device.tap_center()
        print("[adb] tapped center")
