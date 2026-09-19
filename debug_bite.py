import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from autofishing.capture.bluestacks import BlueStacksCapture

CONFIG = Path('config.yaml')
with CONFIG.open('r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cap_cfg = cfg['capture']
model_cfg = cfg['model']

capture = BlueStacksCapture(
    image_width=int(cap_cfg['image_width']),
    image_height=int(cap_cfg['image_height']),
    roi_left=int(cap_cfg['left']),
    roi_top=int(cap_cfg['top']),
    roi_width=int(cap_cfg['width']),
    roi_height=int(cap_cfg['height']),
)

model = YOLO(str(model_cfg['path']))

print('=== BITE DEBUG ===')
print(f"ROI logical: left={cap_cfg['left']} top={cap_cfg['top']} width={cap_cfg['width']} height={cap_cfg['height']}")
print(f"Model: {model_cfg['path']}")
print('Inference conf=0.01, imgsz=640')
print('Khong click game. Hay tu cau bang tay de lam xuat hien dau !.')
print('Nhan Q de thoat. Ctrl+C cung duoc.')

best_conf = -1.0
best_saved_at = 0.0
last_print = 0.0

try:
    while True:
        frame = capture.grab_bgr(full=False)
        frame = np.ascontiguousarray(frame)

        results = model.predict(
            source=frame,
            conf=0.01,
            imgsz=640,
            verbose=False,
        )

        shown = frame.copy()
        ex = []
        other = []

        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                name = str(names[cls_id])
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                item = (conf, name, cx, cy, x1, y1, x2, y2)
                if name.lower() == 'exclamation-mark':
                    ex.append(item)
                else:
                    other.append(item)

        for conf, name, cx, cy, x1, y1, x2, y2 in ex:
            cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                shown,
                f'{name} {conf:.2f}',
                (max(0, x1), max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        now = time.time()
        if ex:
            ex.sort(reverse=True)
            conf, name, cx, cy, *_ = ex[0]
            print(f'[BITE] conf={conf:.3f} center=({cx},{cy})  logical=({cap_cfg["left"] + cx},{cap_cfg["top"] + cy})')
            if conf > best_conf or now - best_saved_at > 1.0:
                cv2.imwrite('debug_bite_best.jpg', shown)
                best_conf = max(best_conf, conf)
                best_saved_at = now
        elif now - last_print >= 2.0:
            print('[BITE] no exclamation-mark above 0.01 on current frame')
            last_print = now

        cv2.putText(
            shown,
            'CONF TEST 0.01 - Q to quit',
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow('bite debug (NO CLICK)', shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
finally:
    capture.close()
    cv2.destroyAllWindows()
