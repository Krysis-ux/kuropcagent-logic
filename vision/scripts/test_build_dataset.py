"""Run: .venv/bin/python scripts/test_build_dataset.py"""
import numpy as np

from build_dataset import is_val, merge, to_yolo


def test_merge_gt_wins_and_dedupes() -> None:
    gt = [[10, 10, 50, 30]]
    dets = [[11, 10, 50, 31, 0.9],     # duplicate of the human box -> dropped
            [100, 100, 140, 120, 0.3],  # new element -> kept
            [200, 200, 240, 220, 0.04],  # below conf -> dropped
            [290, 5, 400, 25, 0.5],     # sticks out of the 300px-wide image -> clipped
            [0, 0, 1, 1, 0.8],          # 1px sliver -> dropped
            [0, 50, 300, 200, 0.15],    # 75% of screen at low score -> dropped
            [0, 50, 300, 200, 0.6]]     # same box, confident -> kept
    out = merge(gt, dets, w=300, h=200, conf=0.05)
    assert out.tolist() == [[10, 10, 50, 30], [100, 100, 140, 120], [290, 5, 300, 25], [0, 50, 300, 200]], out


def test_merge_without_gt_or_dets() -> None:
    assert merge([], [], 100, 100, 0.05).shape == (0, 4)
    assert len(merge([], [[1, 1, 20, 20, 0.5]], 100, 100, 0.05)) == 1


def test_to_yolo_normalizes_center_wh() -> None:
    assert to_yolo(np.array([[0, 0, 50, 100]]), 100, 200) == "0 0.250000 0.250000 0.500000 0.500000\n"
    assert to_yolo(np.array([[0, 0, 50, 100], [0, 0, 50, 100]]), 100, 200).count("\n") == 1  # duplicates collapse


def test_split_is_deterministic_and_about_10pct() -> None:
    vals = [is_val(f"video/{i}") for i in range(5000)]
    assert vals == [is_val(f"video/{i}") for i in range(5000)]
    assert 0.08 < sum(vals) / len(vals) < 0.12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
