from typing import List, Dict, Optional

import numpy as np

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


class YOLOv8Detector:
    def __init__(self, weights: str = "yolov8n.pt"):
        self.weights = weights
        self.model = None

    def name(self) -> str:
        return "yolov8"

    def available(self) -> bool:
        return _HAS_YOLO

    def load(self, cfg: Optional[Dict] = None) -> None:
        if not _HAS_YOLO:
            return
        w = (cfg or {}).get("weights", self.weights)
        self.model = YOLO(w)

    def infer(self, image_bgr) -> List[Dict]:
        if self.model is None or not _HAS_YOLO:
            return []
        res = self.model.predict(source=image_bgr, verbose=False)
        out: List[Dict] = []
        for r in res:
            if getattr(r, "boxes", None) is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                area = w * h
                out.append({
                    "source": self.name(),
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "x_center_px": float(cx),
                    "y_center_px": float(cy),
                    "area_px": float(area),
                    "confidence": conf,
                    "class": cls,
                })
        return out