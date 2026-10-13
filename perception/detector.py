"""Day 10: find objects in the robot's camera picture.

Two methods, combined:
  1. YOLO (pretrained, no training): finds "bottle", "cup", ... but does not know colours.
  2. Colour check: finds red / blue areas. Used to say WHICH bottle is red, and as a
     fallback when YOLO does not recognise the simple simulated cylinders.
"""
import os

import cv2
import numpy as np

# HSV colour ranges (OpenCV hue is 0-180). Red wraps around 0, so it has two ranges.
COLOURS = {
    "red":   [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "blue":  [((100, 120, 70), (130, 255, 255))],
    "green": [((40, 80, 70), (80, 255, 255))],
}
BOX_COLOURS = {"red": (0, 0, 255), "blue": (255, 80, 0), "green": (0, 200, 0)}


def colour_mask(bgr, colour):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOURS[colour]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def find_colour_blobs(bgr, colour, min_area=80):
    """Boxes (x1, y1, x2, y2) around areas of one colour, biggest first."""
    mask = colour_mask(bgr, colour)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def colour_fraction(bgr, box, colour):
    x1, y1, x2, y2 = box
    crop = bgr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    return float(colour_mask(crop, colour).mean() / 255.0)


class ObjectDetector:
    """detect(bgr) -> list of dicts: label, colour, source, conf, box, centre."""

    def __init__(self, model_path="models/yolo11n.pt", conf=0.35, use_yolo=True):
        self.conf = conf
        self.yolo = None
        if use_yolo:
            try:
                from ultralytics import YOLO
                import torch
                self.yolo = YOLO(model_path if os.path.exists(model_path) else "yolo11n.pt")
                self.device = 0 if torch.cuda.is_available() else "cpu"
            except Exception as err:                       # keep working with colour only
                print(f"[detector] YOLO not available ({err}); using colour detection only")

    def _yolo(self, bgr):
        if self.yolo is None:
            return []
        result = self.yolo.predict(bgr, conf=self.conf, device=self.device, verbose=False)[0]
        found = []
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy()):
            found.append({"label": result.names[int(cls)], "conf": float(conf),
                          "box": tuple(int(v) for v in box), "source": "yolo"})
        return found

    def detect(self, bgr):
        detections = self._yolo(bgr)
        # give each YOLO object a colour (e.g. bottle -> red bottle)
        for d in detections:
            d["colour"] = None
            for colour in COLOURS:
                if colour_fraction(bgr, d["box"], colour) > 0.25:
                    d["colour"] = colour
                    break
        # colour fallback: tall coloured blobs that YOLO missed are probably bottles
        for colour in ("red", "blue"):
            for box in find_colour_blobs(bgr, colour):
                if any(_overlap(box, d["box"]) > 0.3 for d in detections):
                    continue
                w, h = box[2] - box[0], box[3] - box[1]
                label = "bottle" if h > 1.4 * w else "object"
                detections.append({"label": label, "colour": colour, "conf": 1.0,
                                   "box": box, "source": "colour"})
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            d["centre"] = ((x1 + x2) // 2, (y1 + y2) // 2)
            d["name"] = f"{d['colour']} {d['label']}" if d["colour"] else d["label"]
        return detections

    @staticmethod
    def find(detections, name):
        """Best match for a request like 'red bottle' (or None)."""
        matches = [d for d in detections if d["name"] == name]
        return max(matches, key=lambda d: d["conf"]) if matches else None


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / area if area else 0.0


def draw(bgr, detections, target=None):
    """Draw boxes and names. The target (e.g. 'red bottle') gets a thick box."""
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        colour = BOX_COLOURS.get(d["colour"], (0, 255, 255))
        thick = 3 if target and d["name"] == target else 1
        cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, thick)
        text = f"{d['name']} {d['conf']:.2f} [{d['source']}]"
        cv2.putText(bgr, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
        cv2.circle(bgr, d["centre"], 3, colour, -1)
    return bgr
