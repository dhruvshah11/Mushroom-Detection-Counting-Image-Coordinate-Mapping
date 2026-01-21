from typing import List, Dict, Optional

import numpy as np

try:
    import torch
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    _HAS_D2 = True
except Exception:
    _HAS_D2 = False


class Detectron2Detector:
    def __init__(self, config_name: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"):
        self.config_name = config_name
        self.predictor = None

    def name(self) -> str:
        return "detectron2"

    def available(self) -> bool:
        return _HAS_D2

    def load(self, cfg_override: Optional[Dict] = None) -> None:
        if not _HAS_D2:
            return
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_name)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float((cfg_override or {}).get("score_thr", 0.25))
        self.predictor = DefaultPredictor(cfg)

    def infer(self, image_bgr) -> List[Dict]:
        if self.predictor is None or not _HAS_D2:
            return []
        outputs = self.predictor(image_bgr)
        inst = outputs["instances"].to("cpu")
        boxes = inst.pred_boxes.tensor.numpy() if hasattr(inst, "pred_boxes") else np.zeros((0, 4), dtype=np.float32)
        scores = inst.scores.numpy() if hasattr(inst, "scores") else np.zeros((0,), dtype=np.float32)
        classes = inst.pred_classes.numpy() if hasattr(inst, "pred_classes") else np.zeros((0,), dtype=np.int64)
        dets: List[Dict] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            area = w * h
            dets.append({
                "source": self.name(),
                "bbox": (int(x1), int(y1), int(w), int(h)),
                "x_center_px": float(cx),
                "y_center_px": float(cy),
                "area_px": float(area),
                "confidence": float(scores[i]),
                "class": int(classes[i]),
            })
        return dets