"""YOLOv8 detection wrapper."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )


class YoloDetector:
    """Thin Ultralytics YOLO wrapper for bite + UI class queries."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.5,
        bite_classes: list[str] | None = None,
        imgsz: int = 320,
    ) -> None:
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Place trained weights at models/best.pt"
            )

        self.model = YOLO(str(path))

        self.confidence = confidence

        self.bite_classes = {
            c.lower()
            for c in (bite_classes or [])
        }

        self.imgsz = imgsz

        # -----------------------------------------------------
        # Bộ nhớ tạm để xác nhận detection ! ở confidence thấp.
        # -----------------------------------------------------

        self._pending_bite: Detection | None = None
        self._pending_bite_at = 0.0

        # Detection confidence từ mức này trở lên
        # có thể chấp nhận ngay nếu hình dạng hợp lệ.
        self._bite_fast_confidence = 0.30

        # Detection thấp hơn 0.30 phải xuất hiện
        # ổn định trên 2 frame liên tiếp.
        self._bite_confirm_window = 0.18

        # Khoảng cách tâm tối đa giữa 2 frame.
        self._bite_confirm_distance = 28

        dummy = np.zeros(
            (
                self.imgsz,
                self.imgsz,
                3,
            ),
            dtype=np.uint8,
        )

        self.model.predict(
            source=dummy,
            conf=0.5,
            imgsz=self.imgsz,
            verbose=False,
        )

    # =========================================================
    # GENERIC DETECTION
    # =========================================================

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        imgsz: int | None = None,
        conf: float | None = None,
    ) -> list[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            conf=(
                self.confidence
                if conf is None
                else conf
            ),
            imgsz=(
                self.imgsz
                if imgsz is None
                else imgsz
            ),
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            names = result.names

            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(
                    box.cls.item()
                )

                class_name = names[
                    cls_id
                ]

                conf_v = float(
                    box.conf.item()
                )

                x1, y1, x2, y2 = (
                    int(v)
                    for v in box.xyxy[0].tolist()
                )

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=conf_v,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )

        return detections

    # =========================================================
    # BITE GEOMETRY FILTER
    # =========================================================

    def _is_plausible_bite(
        self,
        det: Detection,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """
        Lọc false-positive của class exclamation-mark.

        Dấu ! thật trong ROI hiện tại của bạn có đặc điểm
        tương đối ổn định:

            - box hẹp
            - box cao
            - không nằm sát mép ROI
            - nằm trong vùng giữa ROI

        Các vật như:
            - lan can
            - cột
            - mép thuyền
            - cần câu
            - cạnh dài

        thường tạo box nằm sát mép hoặc có kích thước/
        tỷ lệ khác dấu ! thật.
        """

        width = det.x2 - det.x1
        height = det.y2 - det.y1

        if width <= 0 or height <= 0:
            return False

        center_x, center_y = det.center

        # -----------------------------------------------------
        # 1. Không nhận box nằm sát mép ROI.
        # -----------------------------------------------------

        edge_x = 10
        edge_y = 8

        if det.x1 < edge_x:
            return False

        if det.y1 < edge_y:
            return False

        if det.x2 > frame_width - edge_x:
            return False

        if det.y2 > frame_height - edge_y:
            return False

        # -----------------------------------------------------
        # 2. Kích thước box.
        #
        # Các detection ! thật trước đây:
        #
        #   21 x 82
        #   22 x 81
        #   19 x 83
        #   21 x 102
        #
        # -----------------------------------------------------

        min_width = 15
        max_width = 35

        min_height = 55
        max_height = 115

        if width < min_width:
            return False

        if width > max_width:
            return False

        if height < min_height:
            return False

        if height > max_height:
            return False

        # -----------------------------------------------------
        # 3. Tỷ lệ cao / rộng.
        # -----------------------------------------------------

        aspect_ratio = (
            height / width
        )

        min_aspect = 2.5
        max_aspect = 6.5

        if aspect_ratio < min_aspect:
            return False

        if aspect_ratio > max_aspect:
            return False

        # -----------------------------------------------------
        # 4. Vùng vị trí tương đối trong ROI.
        #
        # Các false-positive đã thấy:
        #
        #   (6,7)
        #   (428,25)
        #   (393,59)
        #   (389,33)
        #   (146,50)
        #
        # đều nằm quá sát trên/mép.
        #
        # Các detection ! thật thường nằm ở vùng giữa.
        # -----------------------------------------------------

        if center_x < 60:
            return False

        if center_x > frame_width - 60:
            return False

        if center_y < 70:
            return False

        if center_y > frame_height - 70:
            return False

        return True

    # =========================================================
    # COMPARE TWO BITE DETECTIONS
    # =========================================================

    def _same_bite(
        self,
        first: Detection,
        second: Detection,
    ) -> bool:
        """
        Kiểm tra detection ở 2 frame có phải cùng một dấu !
        """

        x1, y1 = first.center
        x2, y2 = second.center

        distance = (
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        ) ** 0.5

        if (
            distance
            > self._bite_confirm_distance
        ):
            return False

        first_width = (
            first.x2 - first.x1
        )

        first_height = (
            first.y2 - first.y1
        )

        second_width = (
            second.x2 - second.x1
        )

        second_height = (
            second.y2 - second.y1
        )

        # Tránh trường hợp frame sau tự nhiên
        # biến thành một box có kích thước hoàn toàn khác.
        if first_width <= 0:
            return False

        if first_height <= 0:
            return False

        width_ratio = (
            second_width
            / first_width
        )

        height_ratio = (
            second_height
            / first_height
        )

        if (
            width_ratio < 0.55
            or width_ratio > 1.80
        ):
            return False

        if (
            height_ratio < 0.55
            or height_ratio > 1.80
        ):
            return False

        return True

    # =========================================================
    # FIND BITE
    # =========================================================

    def find_bite(
        self,
        frame_bgr: np.ndarray,
    ) -> Detection | None:
        detections = self.detect(
            frame_bgr
        )

        frame_height, frame_width = (
            frame_bgr.shape[:2]
        )

        best: Detection | None = None

        # -----------------------------------------------------
        # Chỉ lấy class exclamation-mark.
        # -----------------------------------------------------

        for det in detections:
            if (
                det.class_name.lower()
                != "exclamation-mark"
            ):
                continue

            # -------------------------------------------------
            # Lọc hình dạng/vị trí.
            # -------------------------------------------------

            if not self._is_plausible_bite(
                det,
                frame_width,
                frame_height,
            ):
                continue

            # -------------------------------------------------
            # Chọn detection có confidence cao nhất.
            # -------------------------------------------------

            if (
                best is None
                or det.confidence
                > best.confidence
            ):
                best = det

        # -----------------------------------------------------
        # Không có candidate hợp lệ.
        # -----------------------------------------------------

        if best is None:
            return None

        now = time.monotonic()

        # =====================================================
        # CONFIDENCE CAO
        # =====================================================

        if (
            best.confidence
            >= self._bite_fast_confidence
        ):
            self._pending_bite = None
            self._pending_bite_at = 0.0

            print(
                f"[debug] ACCEPT ! "
                f"conf={best.confidence:.3f} "
                f"center={best.center} "
                f"box=({best.x1},{best.y1},"
                f"{best.x2},{best.y2})"
            )

            return best

        # =====================================================
        # CONFIDENCE THẤP
        #
        # Không reel ngay.
        # Phải thấy lại detection tương tự ở frame kế tiếp.
        # =====================================================

        if (
            self._pending_bite is not None
            and (
                now
                - self._pending_bite_at
            )
            <= self._bite_confirm_window
        ):
            if self._same_bite(
                self._pending_bite,
                best,
            ):
                # Lấy confidence cao hơn.
                confirmed = (
                    best
                    if best.confidence
                    >= self._pending_bite.confidence
                    else self._pending_bite
                )

                self._pending_bite = None
                self._pending_bite_at = 0.0

                print(
                    f"[debug] ACCEPT ! "
                    f"conf={confirmed.confidence:.3f} "
                    f"center={confirmed.center} "
                    f"box=({confirmed.x1},"
                    f"{confirmed.y1},"
                    f"{confirmed.x2},"
                    f"{confirmed.y2})"
                )

                return confirmed

        # -----------------------------------------------------
        # Lưu candidate để chờ frame kế tiếp.
        # -----------------------------------------------------

        self._pending_bite = best
        self._pending_bite_at = now

        return None

    # =========================================================
    # FIND CLASS
    # =========================================================

    def find_class(
        self,
        frame_bgr: np.ndarray,
        class_names: (
            set[str]
            | list[str]
            | tuple[str, ...]
        ),
        *,
        imgsz: int | None = None,
        conf: float | None = None,
    ) -> Detection | None:
        wanted = {
            c.lower()
            for c in class_names
        }

        best: Detection | None = None

        for det in self.detect(
            frame_bgr,
            imgsz=imgsz,
            conf=conf,
        ):
            if (
                det.class_name.lower()
                in wanted
            ):
                if (
                    best is None
                    or det.confidence
                    > best.confidence
                ):
                    best = det

        return best

    # =========================================================
    # FIND ANY
    # =========================================================

    def find_any(
        self,
        frame_bgr: np.ndarray,
        class_names: (
            set[str]
            | list[str]
            | tuple[str, ...]
        ),
        *,
        imgsz: int | None = None,
        conf: float | None = None,
    ) -> dict[str, Detection]:
        wanted = {
            c.lower()
            for c in class_names
        }

        found: dict[str, Detection] = {}

        for det in self.detect(
            frame_bgr,
            imgsz=imgsz,
            conf=conf,
        ):
            key = det.class_name.lower()

            if key not in wanted:
                continue

            prev = found.get(key)

            if (
                prev is None
                or det.confidence
                > prev.confidence
            ):
                found[key] = det

        return found