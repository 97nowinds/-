"""Compare lab coat and mask models on person crops from an offline replay."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", required=True, type=Path)
    parser.add_argument("--coat-weights", required=True, type=Path)
    parser.add_argument("--mask-vendor", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO
    from ppe_detector import MaskDetector

    coat = YOLO(str(args.coat_weights))
    if set(coat.names.values()) != {"worn", "not_worn"}:
        raise ValueError("Expected lab coat weights with worn and not_worn classes")
    mask = MaskDetector(args.mask_vendor, args.device)
    samples = json.loads((args.crops / "index.json").read_text(encoding="utf-8"))
    seen = set()
    with args.output.open("x", encoding="utf-8") as output:
        for sample in samples:
            sample_id = sample["id"]
            if not isinstance(sample_id, str) or not sample_id.isascii() or not sample_id.isalnum():
                raise ValueError("Invalid replay sample ID")
            if sample_id in seen:
                raise ValueError(f"Duplicate replay sample: {sample_id}")
            seen.add(sample_id)
            crop = cv2.imread(str(args.crops / f"{sample_id}.jpg"))
            if crop is None:
                raise ValueError(f"Cannot read replay crop: {sample_id}")
            result = coat.predict(crop, device=args.device, imgsz=320, verbose=False)[0]
            prediction = {
                **sample,
                "lab_coat": {
                    "state": result.names[result.probs.top1],
                    "score": float(result.probs.top1conf),
                },
                "mask": mask.observe(crop),
                "notification_eligible": False,
            }
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
