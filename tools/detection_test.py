"""Day 10 measurement: run the detector on saved camera pictures and count what it finds.

Usage:  python tools/detection_test.py                 (pictures in data/snapshots/)
        python tools/detection_test.py --target "blue bottle"
Writes pictures with boxes to data/detections/.
"""
import argparse
import collections
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.detector import ObjectDetector, draw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/snapshots/*.png")
ap.add_argument("--target", default="red bottle")
args = ap.parse_args()

files = sorted(glob.glob(args.images))
if not files:
    sys.exit(f"No pictures found in {args.images}. Save some with C in the EYES window first.")

detector = ObjectDetector()
os.makedirs("data/detections", exist_ok=True)
counts, by_source, target_hits = collections.Counter(), collections.Counter(), 0
print(f"{'picture':32s} found")
for path in files:
    img = cv2.imread(path)
    found = detector.detect(img)
    names = [d["name"] for d in found]
    counts.update(set(names))
    by_source.update(d["source"] for d in found)
    hit = ObjectDetector.find(found, args.target)
    target_hits += hit is not None
    cv2.imwrite(os.path.join("data/detections", os.path.basename(path)), draw(img, found, args.target))
    print(f"{os.path.basename(path):32s} {', '.join(names) or '-'}")

n = len(files)
print(f"\n'{args.target}' found in {target_hits} of {n} pictures ({100 * target_hits / n:.0f}%)")
print("Seen in how many pictures:", dict(counts))
print("Found by:", dict(by_source), "(yolo = AI model, colour = colour check)")
print("Pictures with boxes saved in data/detections/")
