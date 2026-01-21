from typing import List, Dict

import numpy as np


def iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
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
    if union <= 0:
        return 0.0
    return float(inter / union)


def nms(dets: List[Dict], iou_thr: float = 0.5) -> List[Dict]:
    if len(dets) == 0:
        return []
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    selected: List[Dict] = []
    for d in dets:
        keep = True
        for s in selected:
            if iou(d["bbox"], s["bbox"]) > iou_thr:
                keep = False
                break
        if keep:
            selected.append(d)
    return selected


def ensemble(dets_list: List[List[Dict]], iou_merge: float = 0.5) -> List[Dict]:
    all_dets = [d for dets in dets_list for d in dets]
    if len(all_dets) == 0:
        return []
    clusters: List[List[Dict]] = []
    for d in sorted(all_dets, key=lambda x: x["confidence"], reverse=True):
        placed = False
        for c in clusters:
            if iou(d["bbox"], c[0]["bbox"]) >= iou_merge:
                c.append(d)
                placed = True
                break
        if not placed:
            clusters.append([d])

    fused: List[Dict] = []
    for c in clusters:
        confs = np.array([x["confidence"] for x in c], dtype=float)
        weights = confs / (confs.sum() + 1e-6)
        xs = np.array([x["x_center_px"] for x in c], dtype=float)
        ys = np.array([x["y_center_px"] for x in c], dtype=float)
        areas = np.array([x["area_px"] for x in c], dtype=float)
        bx = np.array([x["bbox"][0] for x in c], dtype=float)
        by = np.array([x["bbox"][1] for x in c], dtype=float)
        bw = np.array([x["bbox"][2] for x in c], dtype=float)
        bh = np.array([x["bbox"][3] for x in c], dtype=float)
        fused.append({
            "source": ",".join(sorted(list({x.get("source", "unknown") for x in c}))),
            "x_center_px": float((xs * weights).sum()),
            "y_center_px": float((ys * weights).sum()),
            "area_px": float((areas * weights).sum()),
            "bbox": (int((bx * weights).sum()), int((by * weights).sum()), int((bw * weights).sum()), int((bh * weights).sum())),
            "confidence": float(confs.max()),
        })
    return fused