from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False


class SAMSegmenter:
    def __init__(self, checkpoint: Optional[str] = None, model_type: str = "vit_b"):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.predictor = None

    def name(self) -> str:
        return "sam"

    def available(self) -> bool:
        return _HAS_SAM and self.checkpoint is not None

    def load(self, cfg: Optional[Dict] = None) -> None:
        if not self.available():
            return
        ckpt = (cfg or {}).get("checkpoint", self.checkpoint)
        mtype = (cfg or {}).get("model_type", self.model_type)
        sam = sam_model_registry[mtype](checkpoint=ckpt)
        self.predictor = SamPredictor(sam)

    def segment_points(self, image_bgr, points: List[Tuple[float, float]]) -> List[Dict]:
        if self.predictor is None:
            return []
        self.predictor.set_image(image_bgr[:, :, ::-1])
        in_points = np.array(points)
        labels = np.ones((len(points),), dtype=np.int32)
        masks, _, _ = self.predictor.predict(point_coords=in_points, point_labels=labels, multimask_output=False)
        dets: List[Dict] = []
        for mask in masks:
            ys, xs = np.where(mask > 0)
            if xs.size == 0:
                continue
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            w, h = x2 - x1 + 1, y2 - y1 + 1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            area = float(mask.sum())
            dets.append({
                "source": self.name(),
                "bbox": (x1, y1, w, h),
                "x_center_px": cx,
                "y_center_px": cy,
                "area_px": area,
                "confidence": 1.0,
                "mask": mask.astype(np.uint8) * 255,
            })
        return dets