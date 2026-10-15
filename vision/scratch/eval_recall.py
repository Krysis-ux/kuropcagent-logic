"""How many human GT boxes does the detector recover at each confidence cutoff?"""
import sys, io, glob, hashlib, collections, pickle
import numpy as np, pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0, 'scripts')
from detector import Detector
from torchvision.ops import box_iou
import torch

groups = collections.defaultdict(lambda: {'gt': []})  # md5 -> {'bytes', 'gt'}
for p in glob.glob('raw/screenspot/data/*.parquet'):
    for r in pq.read_table(p).to_pylist():
        g = groups[hashlib.md5(r['image']['bytes']).hexdigest()]
        g['bytes'] = r['image']['bytes']; g['src'] = 'screenspot'; g['norm'] = True
        g['gt'].append(r['bbox'])
for r in pq.read_table('raw/waveui/data/train-00000-of-00022.parquet').to_pylist():
    g = groups[hashlib.md5(r['image']['bytes']).hexdigest()]
    g['bytes'] = r['image']['bytes']; g['src'] = g.get('src', 'waveui'); g['norm'] = g.get('norm', False)
    g['gt'].append(r['bbox'] if not g['norm'] else None)
print('images', len(groups))

det = Detector(conf=0.05)
out = []
items = list(groups.values())
for i in range(0, len(items), 8):
    batch = items[i:i+8]
    ims = [Image.open(io.BytesIO(g['bytes'])).convert('RGB') for g in batch]
    for g, im, (bx, sc) in zip(batch, ims, det(ims)):
        w, h = im.size
        gt = np.array([b for b in g['gt'] if b is not None], dtype=np.float32)
        if g['norm']: gt = gt * [w, h, w, h]
        out.append(dict(src=g['src'], wh=(w, h), gt=gt, bx=bx, sc=sc))
    if i % 400 == 0: print(i, flush=True)
pickle.dump(out, open('scratch/recall_raw.pkl', 'wb'))
