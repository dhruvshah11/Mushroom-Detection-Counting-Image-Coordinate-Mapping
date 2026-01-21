from typing import List, Dict, Optional, Tuple

import time
import numpy as np


class Timer:
    def __init__(self):
        self.start_t = None
        self.end_t = None

    def start(self):
        self.start_t = time.perf_counter()

    def stop(self):
        self.end_t = time.perf_counter()

    def fps(self, frames: int) -> float:
        if self.start_t is None or self.end_t is None:
            return 0.0
        elapsed = self.end_t - self.start_t
        if elapsed <= 0:
            return 0.0
        return frames / elapsed


def compute_pr(gt: List[Dict], preds: List[Dict], iou_thr: float = 0.5) -> Dict:
    if gt is None:
        return {"precision": None, "recall": None}
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
    matched_gt = set()
    tp = 0
    fp = 0
    for p in preds:
        pb = p["bbox"]
        ok = False
        for i, g in enumerate(gt):
            if i in matched_gt:
                continue
            if iou(pb, g["bbox"]) >= iou_thr:
                matched_gt.add(i)
                ok = True
                break
        tp += 1 if ok else 0
        fp += 0 if ok else 1
    fn = len(gt) - len(matched_gt)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def compute_pr_aggregate(gt_map: Dict[str, List[Dict]], preds_map: Dict[str, List[Dict]], conf_thr: float, iou_thr: float = 0.5) -> Dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for img_name, gt in gt_map.items():
        preds = [p for p in preds_map.get(img_name, []) if float(p.get("confidence", 1.0)) >= conf_thr]
        r = compute_pr(gt, preds, iou_thr=iou_thr)
        total_tp += int(r.get("tp", 0))
        total_fp += int(r.get("fp", 0))
        total_fn += int(r.get("fn", 0))
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}


def compute_ap(pr_points: List[Tuple[float, float]]) -> float:
    if len(pr_points) == 0:
        return 0.0
    pts = sorted(pr_points, key=lambda x: x[1])
    recalls = np.array([p[1] for p in pts], dtype=float)
    precisions = np.array([p[0] for p in pts], dtype=float)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += precisions[i] * (recalls[i] - recalls[i - 1])
    return float(ap)


def sweep_thresholds(gt_map: Dict[str, List[Dict]], preds_map: Dict[str, List[Dict]], thr_min: float = 0.0, thr_max: float = 1.0, step: float = 0.05, iou_thr: float = 0.5) -> Dict:
    thresholds = []
    t = thr_min
    while t <= thr_max + 1e-9:
        thresholds.append(round(t, 6))
        t += step
    rows = []
    pr_points = []
    best_f1 = -1.0
    best_thr = None
    for thr in thresholds:
        r = compute_pr_aggregate(gt_map, preds_map, conf_thr=thr, iou_thr=iou_thr)
        rows.append({"threshold": thr, "precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})
        pr_points.append((r["precision"], r["recall"]))
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_thr = thr
    ap = compute_ap(pr_points)
    return {"thresholds": thresholds, "rows": rows, "ap": ap, "best_threshold": best_thr, "best_f1": best_f1}