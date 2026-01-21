from dhruv_submission.code.utils.post import iou, nms, ensemble


def test_iou_basic():
    assert abs(iou((0,0,10,10),(5,5,10,10)) - (5*5)/((10*10)+(10*10)-(5*5))) < 1e-6


def test_nms():
    dets = [
        {"bbox": (0,0,10,10), "confidence": 0.9},
        {"bbox": (1,1,10,10), "confidence": 0.8},
        {"bbox": (20,20,5,5), "confidence": 0.7},
    ]
    out = nms(dets, iou_thr=0.5)
    assert len(out) == 2


def test_ensemble():
    d1 = [{"bbox": (0,0,10,10), "confidence": 0.9, "x_center_px": 5, "y_center_px": 5, "area_px": 100}]
    d2 = [{"bbox": (1,1,10,10), "confidence": 0.8, "x_center_px": 6, "y_center_px": 6, "area_px": 100}]
    fused = ensemble([d1, d2], iou_merge=0.5)
    assert len(fused) == 1

if __name__ == "__main__":
    test_iou_basic()
    test_nms()
    test_ensemble()
    print("utils.post tests passed")