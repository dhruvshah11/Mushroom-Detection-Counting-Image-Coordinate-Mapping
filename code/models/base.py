from typing import List, Dict, Optional


class Detector:
    def name(self) -> str:
        return "base"

    def load(self, cfg: Optional[Dict] = None) -> None:
        pass

    def infer(self, image_bgr) -> List[Dict]:
        return []


class Segmenter:
    def name(self) -> str:
        return "base"

    def load(self, cfg: Optional[Dict] = None) -> None:
        pass

    def segment(self, image_bgr) -> List[Dict]:
        return []