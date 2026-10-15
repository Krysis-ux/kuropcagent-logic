"""Draw YOLO labels on random images: .venv/bin/python scripts/preview.py [N] [name-filter] -> dataset/preview/"""
import random
import sys
from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parent.parent / "dataset"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
filt = sys.argv[2] if len(sys.argv) > 2 else ""

labels = [p for p in OUT.glob("labels/*/*.txt") if filt in p.name]
dst = OUT / "preview"
dst.mkdir(exist_ok=True)
for lbl in random.sample(labels, min(n, len(labels))):
    img = cv2.imread(str(lbl.parent.parent.parent / "images" / lbl.parent.name / (lbl.stem + ".jpg")))
    h, w = img.shape[:2]
    t = max(1, round(max(h, w) / 700))
    for line in lbl.read_text().splitlines():
        _, cx, cy, bw, bh = map(float, line.split())
        cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                      (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (255, 0, 255), t)
    cv2.imwrite(str(dst / (lbl.stem + ".jpg")), img)
print(f"wrote {min(n, len(labels))} previews to {dst}")
