"""Build a 1-class YOLO dataset of clickable GUI elements.

stages: extract -> detect (resumable) -> labels (re-run with --conf)
"""
import argparse
import glob
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "raw", ROOT / "dataset"
GT_FILE, DET_FILE = OUT / "gt.json", OUT / "detections.jsonl"
VAL_EVERY = 10  # ~10% val, split by screenshot / by video so near-duplicate frames never cross splits
# WaveUI types that are mostly static content; the detector still picks up the clickable ones.
STATIC_TYPES = {"text", "header", "heading", "title", "label", "image", "banner", "background", "section",
                "separator", "divider", "container", "panel", "display", "text block", "paragraph", "unknown",
                "notification", "notice", "message", "indicator", "badge"}
MIN_SIDE_PX = 2
GT_DEDUPE_IOU = 0.5
# Low-score detections covering a big chunk of the screen are mostly empty lists / whole windows.
# Measured on ScreenSpot+WaveUI human boxes: removes 1.5% of boxes for -0.5pt recall.
BIG_AREA_FRAC, BIG_MIN_CONF = 0.10, 0.2


def is_val(group: str) -> bool:
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % VAL_EVERY == 0


def save_jpg(img_bytes: bytes, path: Path) -> tuple[int, int]:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=95)
    return im.size


# ---------------------------------------------------------------- extract
def extract() -> None:
    gt: dict[str, list] = defaultdict(list)   # image relpath -> [[x1,y1,x2,y2] px]
    by_md5: dict[str, str] = {}                 # dedupe identical screenshots across/within datasets
    sizes: dict[str, tuple[int, int]] = {}

    # ScreenSpot is a public benchmark: keep it entirely in val so training on this set can't leak into it.
    for p in sorted(glob.glob(str(RAW / "screenspot/data/*.parquet"))):
        for r in pq.read_table(p).to_pylist():
            md5 = hashlib.md5(r["image"]["bytes"]).hexdigest()
            if md5 not in by_md5:
                rel = f"images/val/screenspot_{r['data_source']}_{md5[:12]}.jpg"
                by_md5[md5] = rel
                sizes[md5] = save_jpg(r["image"]["bytes"], OUT / rel)
                gt.setdefault(rel, [])
            w, h = sizes[md5]
            x1, y1, x2, y2 = r["bbox"]  # normalized xyxy
            gt[by_md5[md5]].append([x1 * w, y1 * h, x2 * w, y2 * h])
    print(f"screenspot: {len(gt)} images", flush=True)

    n_wave = n_static = 0
    for p in sorted(glob.glob(str(RAW / "waveui/data/*.parquet"))):
        for r in pq.read_table(p).to_pylist():
            md5 = hashlib.md5(r["image"]["bytes"]).hexdigest()
            if md5 not in by_md5:
                split = "val" if r["source"] == "screenspot" or is_val(md5) else "train"
                rel = f"images/{split}/waveui_{r['source']}_{md5[:12]}.jpg"
                by_md5[md5] = rel
                save_jpg(r["image"]["bytes"], OUT / rel)
                gt.setdefault(rel, [])
                n_wave += 1
            if (r["type"] or "").strip().lower() in STATIC_TYPES:
                n_static += 1
                continue
            gt[by_md5[md5]].append(list(r["bbox"]))  # px xyxy
        print(f"  {Path(p).name}: waveui images so far {n_wave}", flush=True)
    print(f"waveui: {n_wave} images, skipped {n_static} static-type boxes", flush=True)

    n_frames = 0
    for f in sorted(glob.glob(str(RAW / "guiworld_frames/*/*.jpg"))):
        plat, stem = Path(f).parent.name, Path(f).stem
        video = stem.rsplit("_", 1)[0]
        split = "val" if is_val(f"{plat}/{video}") else "train"
        dst = OUT / f"images/{split}/guiworld_{plat}_{stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(f, dst)
        n_frames += 1
    print(f"guiworld: {n_frames} frames", flush=True)

    GT_FILE.write_text(json.dumps({k: [[round(v, 1) for v in b] for b in bs] for k, bs in gt.items()}))


# ---------------------------------------------------------------- detect
def detect(batch_size: int) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from detector import Detector

    done = set()
    if DET_FILE.exists():
        done = {json.loads(line)["image"] for line in DET_FILE.open()}
    todo = sorted(str(p.relative_to(OUT)) for p in OUT.glob("images/*/*.jpg"))
    todo = [t for t in todo if t not in done]
    print(f"{len(done)} already done, {len(todo)} to go", flush=True)

    det = Detector(conf=0.05)
    with DET_FILE.open("a") as out:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            ims = [Image.open(OUT / rel).convert("RGB") for rel in batch]
            for rel, im, (bx, sc) in zip(batch, ims, det(ims)):
                boxes = [[*np.round(b, 1).tolist(), round(float(s), 3)] for b, s in zip(bx, sc)]
                out.write(json.dumps({"image": rel, "wh": im.size, "boxes": boxes}) + "\n")
            out.flush()
            if i % (batch_size * 100) == 0:
                print(f"{i + len(batch)}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- labels
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def merge(gt: list, dets: list, w: int, h: int, conf: float) -> np.ndarray:
    """Human boxes win; a detection is added unless it duplicates a human box. Returns clipped xyxy px."""
    g = np.array(gt, dtype=np.float64).reshape(-1, 4)
    big = lambda b: (b[2] - b[0]) * (b[3] - b[1]) > BIG_AREA_FRAC * w * h and b[4] < BIG_MIN_CONF
    d = np.array([b[:4] for b in dets if b[4] >= conf and not big(b)], dtype=np.float64).reshape(-1, 4)
    if len(g) and len(d):
        d = d[iou_matrix(d, g).max(1) < GT_DEDUPE_IOU]
    boxes = np.concatenate([g, d])
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    keep = ((boxes[:, 2] - boxes[:, 0]) >= MIN_SIDE_PX) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_SIDE_PX)
    return boxes[keep]


def to_yolo(boxes: np.ndarray, w: int, h: int) -> str:
    lines = (f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
             for x1, y1, x2, y2 in boxes)
    return "".join(dict.fromkeys(lines))  # WaveUI repeats the same element once per instruction


def labels(conf: float) -> None:
    gt = json.loads(GT_FILE.read_text())
    n_img = n_box = n_gt = 0
    for line in DET_FILE.open():
        r = json.loads(line)
        w, h = r["wh"]
        boxes = merge(gt.get(r["image"], []), r["boxes"], w, h, conf)
        dst = OUT / r["image"].replace("images/", "labels/", 1).replace(".jpg", ".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        txt = to_yolo(boxes, w, h)
        dst.write_text(txt)
        n_img, n_box, n_gt = n_img + 1, n_box + txt.count("\n"), n_gt + len(gt.get(r["image"], []))
    # No `path:` key -> Ultralytics resolves train/val relative to this yaml, so the folder can be moved/uploaded.
    (OUT / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: interactive\n")
    print(f"labels: {n_img} images, {n_box} boxes ({n_box / max(n_img, 1):.1f}/img, {n_gt} human), conf>={conf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "detect", "labels"])
    ap.add_argument("--conf", type=float, default=0.05, help="detector score cutoff for labels stage")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    {"extract": extract, "detect": lambda: detect(a.batch), "labels": lambda: labels(a.conf)}[a.stage]()
