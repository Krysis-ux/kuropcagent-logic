import sys, cv2, time
sys.path.insert(0, 'scripts')
from detector import Detector
from PIL import Image
d = Detector()
ims = [Image.open(f'scratch/ss_{i}.png').convert('RGB') for i in range(4)]
t = time.time(); d(ims); print('warm', time.time() - t)
t = time.time(); res = d(ims); print('batch4', time.time() - t)
for i, (bx, sc) in enumerate(res):
    im = cv2.imread(f'scratch/ss_{i}.png'); s = max(im.shape[:2]) / 1000
    print(i, len(bx))
    for (x1, y1, x2, y2), c in zip(bx.tolist(), sc.tolist()):
        col = (0,200,0) if c >= 0.3 else (0,165,255) if c >= 0.15 else (0,0,255)
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, max(1, int(2*s)))
        cv2.putText(im, f'{c:.2f}', (int(x1)+2, int(y1)+int(18*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*s, col, max(1, int(1.5*s)))
    cv2.imwrite(f'scratch/ss_{i}_v3.jpg', im)
