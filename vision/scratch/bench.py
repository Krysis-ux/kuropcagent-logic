import sys, time, glob, torch, numpy as np
sys.path.insert(0, 'scripts')
from detector import Detector, DEVICE
from PIL import Image
torch.set_grad_enabled(False)
d = Detector()
ims = [Image.open(p).convert('RGB') for p in sorted(glob.glob('raw/guiworld_frames/website/*.jpg'))[:16]]
t = time.time(); pre = [d._letterbox(im) for im in ims]; print('prep/img', (time.time()-t)/16)
x = torch.from_numpy(np.stack([p[0] for p in pre])).to(DEVICE).permute(0,3,1,2).float()/255
for bs in [1, 4, 8]:
    xb = x[:bs]; d.model(xb); torch.mps.synchronize()
    t = time.time(); [d.model(xb) for _ in range(2)]; torch.mps.synchronize(); print('fp32 bs', bs, 'model/img', (time.time()-t)/(2*bs))
mh = d.model.half(); xb = x[:8].half(); mh(xb); torch.mps.synchronize()
t = time.time(); o = [mh(xb) for _ in range(2)]; torch.mps.synchronize(); print('fp16 bs 8 model/img', (time.time()-t)/16)
