import sys, glob, random, torch, numpy as np
sys.path.insert(0, 'scripts')
from detector import Detector
from torchvision.ops import box_iou
from PIL import Image
random.seed(0)
paths = random.sample(glob.glob('raw/guiworld_frames/*/*.jpg'), 24) + glob.glob('scratch/ss_?.png')
ims = [Image.open(p).convert('RGB') for p in paths]
a = Detector(half=False); ra = [r for i in range(0, len(ims), 4) for r in a(ims[i:i+4])]; del a
b = Detector(half=True);  rb = [r for i in range(0, len(ims), 4) for r in b(ims[i:i+4])]
na = nb = matched = 0; sdiff = []
for (ba, sa), (bb, sb) in zip(ra, rb):
    na += len(ba); nb += len(bb)
    if len(ba) and len(bb):
        iou = box_iou(torch.tensor(ba), torch.tensor(bb)); m = iou.max(1)
        matched += (m.values > 0.9).sum().item(); sdiff += list(np.abs(sa[m.indices.numpy()] - sb[m.indices.numpy()])[m.values.numpy() > 0.9])
print(f'fp32 boxes {na}  fp16 boxes {nb}  fp32 boxes matched in fp16 (IoU>0.9): {matched/na:.3f}  mean|dscore| {np.mean(sdiff):.4f}')
