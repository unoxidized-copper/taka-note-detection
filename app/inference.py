from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Union

from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("model", "best.pt"))

# Default confidence threshold for filtering weak detections.
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_PATH}'. "
            "Place best.pt there or set the MODEL_PATH env var."
        )
    return YOLO(MODEL_PATH)


def _load_image(image: Union[str, bytes, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def predict(image: Union[str, bytes, Image.Image],
            conf: float = CONF_THRESHOLD) -> dict:
    pil = _load_image(image)
    model = get_model()

    result = model.predict(pil, conf=conf, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0])
        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": {
                "x_min": round(x1, 2),
                "y_min": round(y1, 2),
                "x_max": round(x2, 2),
                "y_max": round(y2, 2),
            },
        })

    return {
        "count": len(detections),
        "image_size": {"width": pil.width, "height": pil.height},
        "detections": detections,
    }


# cli for test
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(1)

    print(json.dumps(predict(sys.argv[1]), indent=2))