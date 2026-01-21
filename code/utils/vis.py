from typing import List, Dict

import numpy as np
import cv2


def draw_overlay(img, dets: List[Dict], conf_thr: float) -> np.ndarray:
    overlay = img.copy()
    count = 0
    for d in dets:
        if d.get("confidence", 0.0) < conf_thr:
            continue
        count += 1
        x, y, w, h = d["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(overlay, f"conf={d.get('confidence',0):.2f}", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"count={count} thr={conf_thr:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return overlay


def heatmap_from_points(h: int, w: int, dets: List[Dict], sigma: float = 15.0, conf_thr: float = 0.0) -> np.ndarray:
    grid = np.zeros((h, w), dtype=np.float32)
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for d in dets:
        if d.get("confidence", 0.0) < conf_thr:
            continue
        cx = float(d["x_center_px"]) if isinstance(d["x_center_px"], (float, int)) else float(d["x_center_px"])
        cy = float(d["y_center_px"]) if isinstance(d["y_center_px"], (float, int)) else float(d["y_center_px"])
        conf = float(d.get("confidence", 1.0))
        g = np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma * sigma))) * conf
        grid += g
    grid = grid / (grid.max() + 1e-6)
    heat = (grid * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heat_color