# Mushroom CV Pipeline (YOLO/Detectron2/SAM + Classical CV)

## Installation
- Python 3.10+ recommended (tested on Python 3.12)
- Core deps: `pip install numpy opencv-python pillow pillow-heif matplotlib pandas`
- Optional models:
  - YOLOv8: `pip install ultralytics`
  - Detectron2: install per official wheels for your OS/Python/Torch
  - SAM: `pip install segment-anything torch torchvision`
- HEIC support: `pip install pillow-heif`

## Usage
- Run from repository root:
  - `python dhruv_submission/code/run.py`
- Environment options:
  - `MODELS`: comma list of models to use. Default `classic`. Examples:
    - `set MODELS=classic,yolov8`
    - `set MODELS=classic,yolov8,detectron2,sam`
  - `MUSHROOM_CONF_THR`: confidence threshold. Default `0.45`.
  - `YOLO_WEIGHTS`: YOLOv8 weights (e.g., `yolov8n.pt`).
  - `D2_CONFIG`: Detectron2 config (e.g., `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`).
  - `D2_SCORE_THR`: Detectron2 score threshold (default `0.25`).
  - `SAM_CHECKPOINT`: local SAM checkpoint path (required to use SAM), `SAM_MODEL` (e.g., `vit_b`).

## Input/Output
- Inputs read from `Images/` and `Test Images/` directories.
- Outputs written to `dhruv_submission/outputs/`:
  - `overlays/`: annotated images (boxes, counts, mask tint if available)
  - `maps/`: per-image map plots and `combined_map.png`
  - `histograms/`: area histograms
  - `heatmaps/`: detection confidence heatmaps
  - `detections.csv`: `image_name,mushroom_id,x_center_px,y_center_px,area_px,box_w_px,box_h_px,confidence`
  - `counts.csv`: `image_name,count,confidence_threshold`
  - `size_stats.csv`: `image_name,mean_area_px,median_area_px,top5_largest_px`

## Pipeline Overview
- Classical CV: HSV low-saturation + Otsu threshold, morphology, watershed splitting, contour features, per-image confidence calibration.
- Models: YOLOv8 and Detectron2 for detection; SAM for mask refinement (prompted by fused detection centers).
- Post-processing: ensemble fusion (IoU-based clustering, confidence-weighted merging) + NMS.
- Visualizations: overlays, per-image maps, combined map, heatmaps.

## Model Interfaces
- `models/base.py`: defines `Detector`/`Segmenter` base interfaces.
- `models/yolov8.py`: `YOLOv8Detector(weights)` → `load(cfg)`/`infer(image_bgr)`.
- `models/detectron2.py`: `Detectron2Detector(config_name)` → `load(cfg)`/`infer(image_bgr)`.
- `models/sam.py`: `SAMSegmenter(checkpoint, model_type)` → `load(cfg)`/`segment_points(image_bgr, points)`.
- Ensemble/NMS: `utils/post.py`.
- Visualization: `utils/vis.py`.
- Metrics/Timer: `utils/metrics.py`.

## Model Comparison & Metrics
- Inference speed: measured with `utils.metrics.Timer` (FPS = images / elapsed). Integrate this around directory processing if needed.
- Accuracy: Precision/Recall and mAP require ground truth. Provide a JSON with `[{image_name, bbox:[x,y,w,h]}]` per image and supply path in code to compute metrics with `utils.metrics.compute_pr`.
- Without GT, script reports counts and confidence distributions; use threshold sweeps via `MUSHROOM_CONF_THR` to check stability.

## Data Augmentation
- Test-time augmentation (TTA) can be added by flips and multi-scale inference per model; fuse detections via ensemble.
- For training-based augmentation, follow each model’s training docs (not included here).

## Examples
- Classic only: `python dhruv_submission/code/run.py`
- Classic + YOLOv8: `set MODELS=classic,yolov8 && python dhruv_submission/code/run.py`
- Full (requires installs/weights): `set MODELS=classic,yolov8,detectron2,sam && set SAM_CHECKPOINT=C:\path\to\sam_vit_b.pth && python dhruv_submission/code/run.py`

## Tests
- Run utility tests: `python -m dhruv_submission.code.tests.test_post`

## Notes
- Detectron2/SAM require proper Torch versions and GPU/CPU compatibility.
- If a model is unavailable, the pipeline automatically skips it and proceeds with available components.