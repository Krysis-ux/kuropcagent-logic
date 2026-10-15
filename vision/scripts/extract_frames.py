"""Extract distinct screens from GUI-World videos: sample 1 fps, keep a frame only when the screen changed."""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

SRC, DST = Path("raw/guiworld"), Path("raw/guiworld_frames")
MAX_FRAMES = 5          # per video; consecutive frames of one video are near-duplicates for training
CHANGE_THRESH = 6.0     # mean abs diff (0-255) of 64x64 gray thumbnails; cursor moves stay below this
MAX_SIDE = 1920


def thumb(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def extract(video):
    cap = cv2.VideoCapture(str(video))
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) or 1))
    kept, last, idx = [], None, 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                t = thumb(frame)
                if last is None or np.abs(t - last).mean() > CHANGE_THRESH:
                    kept.append((idx, frame))
                    last = t
        idx += 1
    cap.release()
    if not kept:
        return video, 0
    picks = np.linspace(0, len(kept) - 1, min(MAX_FRAMES, len(kept))).round().astype(int)
    out = DST / video.parent.name
    out.mkdir(parents=True, exist_ok=True)
    for i in sorted(set(picks)):
        idx, frame = kept[i]
        h, w = frame.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{video.stem}_{idx:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return video, len(set(picks))


if __name__ == "__main__":
    videos = sorted(p for p in SRC.glob("*/*") if p.suffix.lower() in (".mp4", ".mov"))
    total, empty = 0, []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        for i, (v, n) in enumerate(pool.imap_unordered(extract, videos, chunksize=4)):
            total += n
            if n == 0:
                empty.append(str(v))
            if i % 200 == 0:
                print(f"{i}/{len(videos)} videos, {total} frames", flush=True)
    print(f"done: {len(videos)} videos -> {total} frames; unreadable/empty: {len(empty)} {empty[:5]}")
