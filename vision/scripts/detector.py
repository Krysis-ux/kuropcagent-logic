"""OmniParser v2 `icon_detect_v3` (MIT, YOLOv9-E TorchScript) interactable-region detector.

Decode logic follows the model card: https://huggingface.co/microsoft/OmniParser-v2.0
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

IMGSZ = 1280
STRIDES = (8, 16, 32)
WEIGHTS = Path(__file__).resolve().parent.parent / "models/omniparser/icon_detect_v3/model.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Detector:
    def __init__(self, conf=0.05, iou=0.45, half=DEVICE != "cpu"):
        self.conf, self.iou = conf, iou
        self.dtype = torch.float16 if half else torch.float32  # fp16 ~2x faster on MPS
        self.model = torch.jit.load(str(WEIGHTS), map_location=DEVICE).eval().to(self.dtype)

    def _letterbox(self, image):
        w, h = image.size
        scale = min(IMGSZ / w, IMGSZ / h)
        canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
        canvas.paste(image.resize((round(w * scale), round(h * scale)), Image.BILINEAR), (0, 0))
        return np.asarray(canvas), scale

    @torch.no_grad()
    def __call__(self, images):
        """images: list of RGB PIL images -> list of (boxes xyxy px float ndarray [N,4], scores [N])."""
        prepped = [self._letterbox(im) for im in images]
        x = torch.from_numpy(np.stack([p[0] for p in prepped])).to(DEVICE).permute(0, 3, 1, 2).to(self.dtype) / 255.0
        outputs = self.model(x)

        boxes, scores = [], []
        for i, stride in enumerate(STRIDES):
            cls = outputs[2 * i].float().sigmoid()[:, 0]    # [B, G, G]
            ltrb = outputs[2 * i + 1].float()               # [B, 4, G, G]
            g = cls.shape[-1]
            gy, gx = torch.meshgrid(torch.arange(g, device=x.device), torch.arange(g, device=x.device), indexing="ij")
            cx, cy = gx + 0.5, gy + 0.5
            l, t, r, b = ltrb.unbind(1)
            boxes.append(torch.stack([(cx - l), (cy - t), (cx + r), (cy + b)], -1).reshape(len(images), -1, 4) * stride)
            scores.append(cls.reshape(len(images), -1))
        boxes, scores = torch.cat(boxes, 1).cpu(), torch.cat(scores, 1).cpu()

        results = []
        for (_, scale), im, bx, sc in zip(prepped, images, boxes, scores):
            keep = sc > self.conf
            bx, sc = bx[keep], sc[keep]
            keep = nms(bx, sc, self.iou)
            bx, sc = bx[keep] / scale, sc[keep]
            w, h = im.size
            bx[:, 0::2] = bx[:, 0::2].clamp(0, w)
            bx[:, 1::2] = bx[:, 1::2].clamp(0, h)
            results.append((bx.numpy(), sc.numpy()))
        return results
