"""Download a platform-balanced random subset of ONE-Lab/GUI-World videos (~7.8 GB)."""
import random

from huggingface_hub import HfApi, snapshot_download

REPO = "ONE-Lab/GUI-World"
# GB per platform; full dataset is 94 GB, total download budget for this project is ~20 GB.
BUDGET_GB = {"android": 0.3, "XR": 0.61, "IOS": 1.6, "website": 1.8, "software": 2.0, "multi": 1.5}

files = [f for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True)
         if getattr(f, "size", None) and f.path.endswith((".mp4", ".mov"))]
random.seed(0)
random.shuffle(files)

picked, used = [], dict.fromkeys(BUDGET_GB, 0)
for f in files:
    plat = f.path.split("/")[0]
    if used[plat] + f.size <= BUDGET_GB[plat] * 1e9:
        picked.append(f.path)
        used[plat] += f.size

for p, b in used.items():
    print(f"{p:9s} {sum(x.startswith(p + '/') for x in picked):5d} videos {b / 1e9:.2f} GB", flush=True)
snapshot_download(REPO, repo_type="dataset", local_dir="raw/guiworld",
                  allow_patterns=picked + ["Annotation/*"], max_workers=8)
print("done", flush=True)
