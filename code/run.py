import os
import sys
from typing import List, Dict, Tuple

import numpy as np
import cv2
from PIL import Image

try:
    from pillow_heif import open_heif
    _HAS_HEIF = True
except Exception:
    _HAS_HEIF = False

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .utils.post import nms, ensemble
    from .utils.vis import draw_overlay, heatmap_from_points
    from .utils.metrics import Timer
    from .models.yolov8 import YOLOv8Detector
    from .models.detectron2 import Detectron2Detector
    from .models.sam import SAMSegmenter
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    from utils.post import nms, ensemble
    from utils.vis import draw_overlay, heatmap_from_points
    from utils.metrics import Timer
    from models.yolov8 import YOLOv8Detector
    from models.detectron2 import Detectron2Detector
    from models.sam import SAMSegmenter


def ensure_dirs(base_out: str) -> Dict[str, str]:
    overlays = os.path.join(base_out, "overlays")
    maps = os.path.join(base_out, "maps")
    histograms = os.path.join(base_out, "histograms")
    os.makedirs(overlays, exist_ok=True)
    os.makedirs(maps, exist_ok=True)
    os.makedirs(histograms, exist_ok=True)
    logs = os.path.join(base_out, "logs")
    os.makedirs(logs, exist_ok=True)
    return {"overlays": overlays, "maps": maps, "histograms": histograms, "logs": logs}


def read_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".heic", ".heif"] and _HAS_HEIF:
        heif_file = open_heif(path)
        image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None and ext in [".heic", ".heif"] and not _HAS_HEIF:
            raise RuntimeError("HEIC reading requires pillow-heif. Please install it.")
        return img


def detect_mushrooms(img: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s_thr = 60
    low_sat = cv2.inRange(s, 0, s_thr)

    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cand = cv2.bitwise_and(thr, low_sat)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_norm, 0.4, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)
    sure_bg = cv2.dilate(clean, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    ws_img = img.copy()
    cv2.watershed(ws_img, markers)

    detections: List[Dict] = []
    overlay_mask = np.zeros((h, w), dtype=np.uint8)

    min_area = max(50, int(0.0001 * w * h))
    max_area = int(0.25 * w * h)

    labels = np.unique(markers)
    for label in labels:
        if label <= 1:
            continue
        seg = (markers == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, bw, bh = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                cx, cy = x + bw / 2.0, y + bh / 2.0
            else:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

            roi_x1 = max(0, x - 10)
            roi_y1 = max(0, y - 10)
            roi_x2 = min(w, x + bw + 10)
            roi_y2 = min(h, y + bh + 10)
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_mask = np.zeros_like(roi, dtype=np.uint8)
            cv2.drawContours(roi_mask, [cnt - [roi_x1, roi_y1]], -1, 255, thickness=-1)
            if roi.size > 0:
                mask_pixels = roi[roi_mask == 255]
                bg_pixels = roi[roi_mask == 0]
                if mask_pixels.size == 0 or bg_pixels.size == 0:
                    contrast = 0.0
                else:
                    contrast = float(abs(float(mask_pixels.mean()) - float(bg_pixels.mean())) / 255.0)
            else:
                contrast = 0.0
            conf = float(np.clip(0.7 * circularity + 0.3 * contrast, 0.0, 1.0))
            cv2.drawContours(overlay_mask, [cnt], -1, 255, thickness=-1)
            detections.append({
                "id": len(detections) + 1,
                "x_center_px": float(cx),
                "y_center_px": float(cy),
                "area_px": float(area),
                "box_w_px": int(bw),
                "box_h_px": int(bh),
                "confidence": conf,
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "circularity": float(circularity),
                "contour": cnt,
            })

    if len(detections) > 0:
        confs = np.array([d["confidence"] for d in detections], dtype=float)
        q1 = np.quantile(confs, 0.25)
        q3 = np.quantile(confs, 0.75)
        scale = float(max(q3 - q1, 1e-6))
        for d in detections:
            d["confidence"] = float(np.clip((d["confidence"] - q1) / scale, 0.0, 1.0))

    return detections, overlay_mask


def draw_overlays(img: np.ndarray, detections: List[Dict], mask: np.ndarray, conf_thr: float) -> np.ndarray:
    overlay = img.copy()
    color_mask = np.zeros_like(img)
    color_mask[:, :, 1] = mask
    alpha = 0.25
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)

    base_overlay = draw_overlay(overlay, detections, conf_thr)
    return base_overlay


def save_map_plot(image_name: str, detections: List[Dict], w: int, h: int, out_path: str, conf_thr: float) -> None:
    xs = []
    ys = []
    sizes = []
    for det in detections:
        if det["confidence"] < conf_thr:
            continue
        xs.append(det["x_center_px"] / float(w))
        ys.append(det["y_center_px"] / float(h))
        sizes.append(det["area_px"])

    if len(xs) == 0:
        xs = [0.5]
        ys = [0.5]
        sizes = [1.0]

    sizes_norm = np.array(sizes)
    if sizes_norm.max() > 0:
        sizes_norm = 200.0 * (sizes_norm / sizes_norm.max())
    else:
        sizes_norm = np.ones_like(sizes_norm) * 50.0

    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, s=sizes_norm, c="tab:green", alpha=0.7, edgecolors="k")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x_norm")
    plt.ylabel("y_norm")
    plt.title(f"Map: {image_name}")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_histogram(image_name: str, detections: List[Dict], out_path: str, conf_thr: float) -> Tuple[float, float, List[float]]:
    areas = [d["area_px"] for d in detections if d["confidence"] >= conf_thr]
    if len(areas) == 0:
        areas = [0.0]
    plt.figure(figsize=(6, 4))
    plt.hist(areas, bins=max(5, int(np.sqrt(len(areas)))), color="tab:blue", alpha=0.7)
    plt.xlabel("mask area (px)")
    plt.ylabel("count")
    plt.title(f"Size histogram: {image_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    mean_area = float(np.mean(areas))
    median_area = float(np.median(areas))
    top5 = list(sorted(areas, reverse=True))[:5]
    return mean_area, median_area, top5


def process_directory(in_dir: str, out_dirs: Dict[str, str], detections_rows: List[Dict], counts_rows: List[Dict], stats_rows: List[Dict], conf_thr: float, combined_points: List[Tuple[float, float, float, str]], models: List[str], yolo: YOLOv8Detector, d2: Detectron2Detector, sam: SAMSegmenter, model_counts_rows: List[Dict], tta_opts: List[str], limit_images: int = 0) -> int:
    entries = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    entries.sort()
    processed = 0
    for fname in entries:
        fpath = os.path.join(in_dir, fname)
        try:
            img = read_image(fpath)
            if img is None:
                continue
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
            continue

        classical_detections, mask = detect_mushrooms(img)
        model_dets: List[List[Dict]] = []
        for m in models:
            if m == "classic":
                model_dets.append([{**d, "source": "classic"} for d in classical_detections])
            elif m == "yolov8":
                if yolo and yolo.available():
                    yolo_preds = yolo.infer(img)
                    dets_aug = list(yolo_preds)
                    if "flip" in tta_opts:
                        img_fl = cv2.flip(img, 1)
                        preds_fl = yolo.infer(img_fl)
                        h0, w0 = img.shape[:2]
                        for d in preds_fl:
                            x, y, bw, bh = d["bbox"]
                            d["bbox"] = (int(w0 - (x + bw)), int(y), int(bw), int(bh))
                            d["x_center_px"] = float(w0 - d["x_center_px"])  # mirror center
                        dets_aug.extend(preds_fl)
                    if "scale" in tta_opts:
                        for s in [0.75, 1.25]:
                            h0, w0 = img.shape[:2]
                            img_rs = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_LINEAR)
                            preds_rs = yolo.infer(img_rs)
                            for d in preds_rs:
                                x, y, bw, bh = d["bbox"]
                                d["bbox"] = (int(x / s), int(y / s), int(bw / s), int(bh / s))
                                d["x_center_px"] = float(d["x_center_px"] / s)
                                d["y_center_px"] = float(d["y_center_px"] / s)
                                d["area_px"] = float(d["area_px"] / (s * s))
                            dets_aug.extend(preds_rs)
                    model_dets.append(dets_aug)
            elif m == "detectron2":
                if d2 and d2.available():
                    d2_preds = d2.infer(img)
                    dets_aug = list(d2_preds)
                    if "flip" in tta_opts:
                        img_fl = cv2.flip(img, 1)
                        preds_fl = d2.infer(img_fl)
                        h0, w0 = img.shape[:2]
                        for d in preds_fl:
                            x, y, bw, bh = d["bbox"]
                            d["bbox"] = (int(w0 - (x + bw)), int(y), int(bw), int(bh))
                            d["x_center_px"] = float(w0 - d["x_center_px"])  # mirror center
                        dets_aug.extend(preds_fl)
                    if "scale" in tta_opts:
                        for s in [0.75, 1.25]:
                            h0, w0 = img.shape[:2]
                            img_rs = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_LINEAR)
                            preds_rs = d2.infer(img_rs)
                            for d in preds_rs:
                                x, y, bw, bh = d["bbox"]
                                d["bbox"] = (int(x / s), int(y / s), int(bw / s), int(bh / s))
                                d["x_center_px"] = float(d["x_center_px"] / s)
                                d["y_center_px"] = float(d["y_center_px"] / s)
                                d["area_px"] = float(d["area_px"] / (s * s))
                            dets_aug.extend(preds_rs)
                    model_dets.append(dets_aug)
            else:
                pass
        for dets, m in zip(model_dets, models):
            kept_m = [d for d in dets if d.get("confidence", 0.0) >= conf_thr]
            model_counts_rows.append({"image_name": fname, "model": m, "count": len(kept_m), "confidence_threshold": conf_thr})
        fused = ensemble(model_dets, iou_merge=0.5)
        fused = nms(fused, iou_thr=0.5)

        if "sam" in models and sam and sam.available() and len(fused) > 0:
            pts = [(d["x_center_px"], d["y_center_px"]) for d in fused]
            sam_dets = sam.segment_points(img, pts)
            if len(sam_dets) > 0:
                fused = nms(ensemble([fused, sam_dets], iou_merge=0.5), iou_thr=0.5)
        h, w = img.shape[:2]
        if any("mask" in d for d in fused):
            union_mask = np.zeros((h, w), dtype=np.uint8)
            for d in fused:
                if "mask" in d and isinstance(d["mask"], np.ndarray):
                    union_mask = cv2.bitwise_or(union_mask, d["mask"])
        else:
            union_mask = mask
        overlay = draw_overlays(img, fused, union_mask, conf_thr)

        overlay_out = os.path.join(out_dirs["overlays"], f"{os.path.splitext(fname)[0]}_overlay.jpg")
        cv2.imwrite(overlay_out, overlay)

        map_out = os.path.join(out_dirs["maps"], f"{os.path.splitext(fname)[0]}_map.png")
        save_map_plot(fname, fused, w, h, map_out, conf_thr)

        hist_out = os.path.join(out_dirs["histograms"], f"{os.path.splitext(fname)[0]}_hist.png")
        mean_area, median_area, top5 = save_histogram(fname, fused, hist_out, conf_thr)

        kept = [d for d in fused if d["confidence"] >= conf_thr]
        counts_rows.append({"image_name": fname, "count": len(kept), "confidence_threshold": conf_thr})

        for d in kept:
            detections_rows.append({
                "image_name": fname,
                "mushroom_id": d.get("id", 0),
                "x_center_px": d["x_center_px"],
                "y_center_px": d["y_center_px"],
                "area_px": d["area_px"],
                "box_w_px": d["bbox"][2],
                "box_h_px": d["bbox"][3],
                "confidence": d["confidence"],
            })
            x_norm = d["x_center_px"] / float(w)
            y_norm = d["y_center_px"] / float(h)
            combined_points.append((x_norm, y_norm, d["area_px"], fname))

        heatmap = heatmap_from_points(h, w, fused, sigma=15.0, conf_thr=conf_thr)
        heat_out = os.path.join(out_dirs.get("heatmaps", out_dirs["maps"]), f"{os.path.splitext(fname)[0]}_heatmap.png")
        cv2.imwrite(heat_out, heatmap)

        try:
            import json
            log_path = os.path.join(out_dirs["logs"], f"{os.path.splitext(fname)[0]}_detections.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([{k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in {
                    "source": d.get("source"),
                    "bbox": d.get("bbox"),
                    "x_center_px": d.get("x_center_px"),
                    "y_center_px": d.get("y_center_px"),
                    "area_px": d.get("area_px"),
                    "confidence": d.get("confidence"),
                }.items()} for d in fused], f, indent=2)
        except Exception:
            pass

        stats_rows.append({
            "image_name": fname,
            "mean_area_px": mean_area,
            "median_area_px": median_area,
            "top5_largest_px": ";".join([f"{a:.1f}" for a in top5]),
        })
        processed += 1
        if limit_images and processed >= limit_images:
            break
    return processed


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    submission_root = os.path.abspath(os.path.join(base_dir, os.pardir))
    outputs_dir = os.path.join(submission_root, "outputs")
    out_dirs = ensure_dirs(outputs_dir)
    heatmaps_dir = os.path.join(outputs_dir, "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)
    out_dirs["heatmaps"] = heatmaps_dir

    images_dir = os.path.abspath(os.path.join(submission_root, os.pardir, "Images"))
    tests_dir = os.path.abspath(os.path.join(submission_root, os.pardir, "Test Images"))

    conf_thr = float(os.environ.get("MUSHROOM_CONF_THR", "0.45"))
    models_spec = os.environ.get("MODELS", "classic")
    models = [m.strip() for m in models_spec.split(",") if m.strip()]
    tta_spec = os.environ.get("USE_TTA", "")
    tta_opts = [t.strip() for t in tta_spec.split(",") if t.strip()]
    yolo_weights = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
    yolo = YOLOv8Detector(weights=yolo_weights)
    if "yolov8" in models and yolo.available():
        yolo.load({"weights": yolo_weights})

    d2_config = os.environ.get("D2_CONFIG", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    d2 = Detectron2Detector(config_name=d2_config)
    if "detectron2" in models and d2.available():
        d2.load({"score_thr": float(os.environ.get("D2_SCORE_THR", "0.25"))})

    sam_ckpt = os.environ.get("SAM_CHECKPOINT")
    sam = SAMSegmenter(checkpoint=sam_ckpt, model_type=os.environ.get("SAM_MODEL", "vit_b"))
    if "sam" in models and sam.available():
        sam.load({"checkpoint": sam_ckpt, "model_type": os.environ.get("SAM_MODEL", "vit_b")})

    detections_rows: List[Dict] = []
    counts_rows: List[Dict] = []
    stats_rows: List[Dict] = []
    combined_points: List[Tuple[float, float, float, str]] = []
    model_counts_rows: List[Dict] = []

    timer = Timer()
    timer.start()
    frames = 0
    limit_images = int(os.environ.get("LIMIT_IMAGES", "0") or 0)
    if os.path.isdir(images_dir):
        frames += process_directory(images_dir, out_dirs, detections_rows, counts_rows, stats_rows, conf_thr, combined_points, models, yolo, d2, sam, model_counts_rows, tta_opts, limit_images)
    if os.path.isdir(tests_dir):
        frames += process_directory(tests_dir, out_dirs, detections_rows, counts_rows, stats_rows, conf_thr, combined_points, models, yolo, d2, sam, model_counts_rows, tta_opts, limit_images)
    timer.stop()

    det_df = pd.DataFrame(detections_rows, columns=[
        "image_name", "mushroom_id", "x_center_px", "y_center_px", "area_px", "box_w_px", "box_h_px", "confidence"
    ])
    det_csv = os.path.join(outputs_dir, "detections.csv")
    det_df.to_csv(det_csv, index=False)

    counts_df = pd.DataFrame(counts_rows, columns=["image_name", "count", "confidence_threshold"])
    counts_csv = os.path.join(outputs_dir, "counts.csv")
    counts_df.to_csv(counts_csv, index=False)

    stats_df = pd.DataFrame(stats_rows, columns=["image_name", "mean_area_px", "median_area_px", "top5_largest_px"])
    stats_csv = os.path.join(outputs_dir, "size_stats.csv")
    stats_df.to_csv(stats_csv, index=False)

    combined_map = os.path.join(out_dirs["maps"], "combined_map.png")
    if len(combined_points) > 0:
        xs = [p[0] for p in combined_points]
        ys = [p[1] for p in combined_points]
        sizes = np.array([p[2] for p in combined_points], dtype=float)
        if sizes.max() > 0:
            sizes_norm = 200.0 * (sizes / sizes.max())
        else:
            sizes_norm = np.ones_like(sizes) * 50.0
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, s=sizes_norm, c="tab:green", alpha=0.6, edgecolors="k")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("x_norm")
        plt.ylabel("y_norm")
        plt.title("Combined Map")
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        plt.savefig(combined_map)
        plt.close()
    else:
        plt.figure(figsize=(6, 6))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("x_norm")
        plt.ylabel("y_norm")
        plt.title("Combined Map")
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()
        plt.savefig(combined_map)
        plt.close()

    gt_path = os.environ.get("GT_JSON")
    metrics_csv = os.path.join(outputs_dir, "metrics.csv")
    metrics_dir = os.path.join(outputs_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    if gt_path and os.path.isfile(gt_path):
        import json
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_items = json.load(f)
        gt_map: Dict[str, List[Dict]] = {}
        for item in gt_items:
            gt_map.setdefault(item["image_name"], []).append({"bbox": tuple(item["bbox"])})
        det_df = pd.read_csv(det_csv)
        preds_fused_map: Dict[str, List[Dict]] = {}
        for img_name, group in det_df.groupby("image_name"):
            preds_fused_map[img_name] = [{"bbox": (int(row["x_center_px"] - row["box_w_px"] / 2.0), int(row["y_center_px"] - row["box_h_px"] / 2.0), int(row["box_w_px"]), int(row["box_h_px"])), "confidence": float(row["confidence"]) } for _, row in group.iterrows()]
        from .utils.metrics import sweep_thresholds
        sw_fused = sweep_thresholds(gt_map, preds_fused_map, thr_min=float(os.environ.get("SWEEP_MIN", "0.0")), thr_max=float(os.environ.get("SWEEP_MAX", "1.0")), step=float(os.environ.get("SWEEP_STEP", "0.05")), iou_thr=0.5)
        pd.DataFrame(sw_fused["rows"]).to_csv(os.path.join(metrics_dir, "sweep_fused.csv"), index=False)
        pd.DataFrame([{"ap": sw_fused["ap"], "best_threshold": sw_fused["best_threshold"], "best_f1": sw_fused["best_f1"]}]).to_csv(os.path.join(metrics_dir, "summary_fused.csv"), index=False)
        try:
            rs = sw_fused["rows"]
            thr = [r["threshold"] for r in rs]
            prec = [r["precision"] for r in rs]
            rec = [r["recall"] for r in rs]
            f1 = [r["f1"] for r in rs]
            plt.figure(figsize=(6,5)); plt.plot(rec, prec); plt.xlabel("recall"); plt.ylabel("precision"); plt.title("PR (fused)"); plt.tight_layout(); plt.savefig(os.path.join(metrics_dir, "pr_fused.png")); plt.close()
            plt.figure(figsize=(6,4)); plt.plot(thr, prec); plt.xlabel("threshold"); plt.ylabel("precision"); plt.title("Precision vs threshold (fused)"); plt.tight_layout(); plt.savefig(os.path.join(metrics_dir, "prec_thr_fused.png")); plt.close()
            plt.figure(figsize=(6,4)); plt.plot(thr, rec); plt.xlabel("threshold"); plt.ylabel("recall"); plt.title("Recall vs threshold (fused)"); plt.tight_layout(); plt.savefig(os.path.join(metrics_dir, "rec_thr_fused.png")); plt.close()
            plt.figure(figsize=(6,4)); plt.plot(thr, f1); plt.xlabel("threshold"); plt.ylabel("f1"); plt.title("F1 vs threshold (fused)"); plt.tight_layout(); plt.savefig(os.path.join(metrics_dir, "f1_thr_fused.png")); plt.close()
        except Exception:
            pass
        model_counts_df = pd.read_csv(os.path.join(outputs_dir, "model_counts.csv"))
        models_unique = sorted(list(set(model_counts_df["model"].tolist())))
        logs_dir = os.path.join(outputs_dir, "logs")
        for m in models_unique:
            preds_map_m: Dict[str, List[Dict]] = {}
            for f in os.listdir(logs_dir):
                if not f.endswith("_detections.json"):
                    continue
                p = os.path.join(logs_dir, f)
                import json
                with open(p, "r", encoding="utf-8") as fh:
                    dets = json.load(fh)
                img_name = f.replace("_detections.json", "").replace("_", ".").split(".")[0]
                preds_map_m.setdefault(f.replace("_detections.json", ""), [])
                preds_map_m[f.replace("_detections.json", "")] = [ {"bbox": d["bbox"], "confidence": float(d.get("confidence", 1.0)) } for d in dets if d.get("source", "")==m ]
            sw_m = sweep_thresholds(gt_map, preds_map_m, thr_min=float(os.environ.get("SWEEP_MIN", "0.0")), thr_max=float(os.environ.get("SWEEP_MAX", "1.0")), step=float(os.environ.get("SWEEP_STEP", "0.05")), iou_thr=0.5)
            pd.DataFrame(sw_m["rows"]).to_csv(os.path.join(metrics_dir, f"sweep_{m}.csv"), index=False)
            pd.DataFrame([{"ap": sw_m["ap"], "best_threshold": sw_m["best_threshold"], "best_f1": sw_m["best_f1"]}]).to_csv(os.path.join(metrics_dir, f"summary_{m}.csv"), index=False)
            try:
                rs = sw_m["rows"]
                thr = [r["threshold"] for r in rs]
                prec = [r["precision"] for r in rs]
                rec = [r["recall"] for r in rs]
                f1 = [r["f1"] for r in rs]
                plt.figure(figsize=(6,5)); plt.plot(rec, prec); plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"PR ({m})"); plt.tight_layout(); plt.savefig(os.path.join(metrics_dir, f"pr_{m}.png")); plt.close()
                plt.figure(figsize=(6,4)); plt.plot(thr, prec); plt.xlabel("threshold"); plt.ylabel("precision"); plt.title(f"Precision vs threshold ({m})"); plt.tight_layout(); plt.savefig(os.path.join(metrics_dir, f"prec_thr_{m}.png")); plt.close()
                plt.figure(figsize=(6,4)); plt.plot(thr, rec); plt.xlabel("threshold"); plt.ylabel("recall"); plt.title(f"Recall vs threshold ({m})"); plt.tight_layout(); plt.savefig(os.path.join(metrics_dir, f"rec_thr_{m}.png")); plt.close()
                plt.figure(figsize=(6,4)); plt.plot(thr, f1); plt.xlabel("threshold"); plt.ylabel("f1"); plt.title(f"F1 vs threshold ({m})"); plt.tight_layout(); plt.savefig(os.path.join(metrics_dir, f"f1_thr_{m}.png")); plt.close()
            except Exception:
                pass

    model_counts_df = pd.DataFrame(model_counts_rows, columns=["image_name", "model", "count", "confidence_threshold"])
    model_counts_csv = os.path.join(outputs_dir, "model_counts.csv")
    model_counts_df.to_csv(model_counts_csv, index=False)

    fps_csv = os.path.join(outputs_dir, "metrics_fps.csv")
    pd.DataFrame([{"images": frames, "fps": timer.fps(max(frames, 1))}]).to_csv(fps_csv, index=False)

    print(f"Wrote: {det_csv}")
    print(f"Wrote: {counts_csv}")
    print(f"Wrote: {stats_csv}")
    if gt_path and os.path.isfile(gt_path):
        print(f"Wrote: {metrics_csv}")
        print(f"Sweep and plots: {metrics_dir}")
    print(f"Wrote: {model_counts_csv}")
    print(f"Wrote: {fps_csv}")
    print(f"Overlays: {out_dirs['overlays']}")
    print(f"Maps: {out_dirs['maps']}")
    print(f"Histograms: {out_dirs['histograms']}")
    print(f"Logs: {out_dirs['logs']}")


if __name__ == "__main__":
    main()