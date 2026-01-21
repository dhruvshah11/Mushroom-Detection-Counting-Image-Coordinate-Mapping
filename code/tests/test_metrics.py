from dhruv_submission.code.utils.metrics import compute_pr_aggregate, sweep_thresholds


def small_maps():
    gt = {
        "img1": [{"bbox": (0, 0, 10, 10)}],
        "img2": [{"bbox": (20, 20, 10, 10)}],
    }
    preds = {
        "img1": [
            {"bbox": (0, 0, 10, 10), "confidence": 0.9},
            {"bbox": (30, 30, 5, 5), "confidence": 0.2},
        ],
        "img2": [
            {"bbox": (20, 20, 10, 10), "confidence": 0.6},
            {"bbox": (100, 100, 5, 5), "confidence": 0.1},
        ],
    }
    return gt, preds


def test_compute_pr_aggregate():
    gt, preds = small_maps()
    r = compute_pr_aggregate(gt, preds, conf_thr=0.5, iou_thr=0.5)
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0
    assert abs(r["precision"] - 1.0) < 1e-6
    assert abs(r["recall"] - 1.0) < 1e-6


def test_sweep_thresholds():
    gt, preds = small_maps()
    sw = sweep_thresholds(gt, preds, thr_min=0.0, thr_max=1.0, step=0.5, iou_thr=0.5)
    assert "ap" in sw and "rows" in sw and len(sw["rows"]) >= 3
    assert sw["best_threshold"] is not None


if __name__ == "__main__":
    test_compute_pr_aggregate()
    test_sweep_thresholds()
    print("utils.metrics tests passed")