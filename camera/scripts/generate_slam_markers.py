"""Generate printable ArUco anchors for the laboratory SLAM setup."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Generate ArUco SLAM marker PNGs")
    parser.add_argument("--output", default="data/slam_markers")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--ids", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--size", type=int, default=800)
    args = parser.parse_args()

    dictionary_id = getattr(cv2.aruco, args.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for marker_id in args.ids:
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, args.size)
        path = output / f"aruco_{args.dictionary}_{marker_id}.png"
        if not cv2.imwrite(str(path), image):
            raise SystemExit(f"无法写入 {path}")
        print(f"marker {marker_id}: {path}")


if __name__ == "__main__":
    main()
