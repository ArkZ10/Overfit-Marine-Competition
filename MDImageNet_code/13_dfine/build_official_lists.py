#!/usr/bin/env python3
"""Map the official competition train set onto our on-disk image files.

The official train_dataset.zip contains the SAME photographs as MDImageDataset,
byte-identical in content and resolution, only renamed to hex ids (verified:
300/300 phash distance 0, identical resolutions, identical boxes). So instead of
extracting 19 GB we identify which of our files are official members by
perceptual hash, and write list files pointing at the images already on disk.

Outputs (under 13_dfine/lists/):
  official_train.txt  official members MINUS anything in our current val split
  official_val.txt    our existing val split (unchanged, so D-FINE stays
                      directly comparable and fusable with detectors A/B/C)
  membership.json     stats + the hex->our-filename map
"""
import json, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ZIP = Path('/root/Overfit-Marine-Competition/MDImageDataset2/train_dataset.zip')
SPLIT = Path('/root/Overfit-Marine-Competition/MDImageDataset/yolo_split')
OUT = Path(__file__).resolve().parent / 'lists'
_z = None

def zph(name):
    global _z
    import imagehash, io
    from PIL import Image
    if _z is None: _z = zipfile.ZipFile(ZIP)
    try:
        with Image.open(io.BytesIO(_z.read(name))) as im:
            return int(str(imagehash.phash(im)), 16)
    except Exception:
        return None

def dph(p):
    import imagehash
    from PIL import Image
    try:
        with Image.open(p) as im: return int(str(imagehash.phash(im)), 16)
    except Exception:
        return None

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    names = [i.filename for i in z.infolist()
             if not i.is_dir() and i.filename.lower().endswith(('.jpg', '.jpeg'))]
    train_imgs = sorted((SPLIT/'images'/'train').glob('*.jpg'))
    val_imgs   = sorted((SPLIT/'images'/'val').glob('*.jpg'))
    ours = train_imgs + val_imgs
    print(f'hashing {len(names)} official + {len(ours)} ours ...', flush=True)
    with ProcessPoolExecutor(max_workers=16) as ex:
        hz = list(ex.map(zph, names, chunksize=32))
        ho = list(ex.map(dph, [str(p) for p in ours], chunksize=64))

    official = {h for h in hz if h is not None}
    hexmap = {}
    for n, h in zip(names, hz):
        if h is not None: hexmap.setdefault(h, Path(n).stem)

    val_stems = {p.stem for p in val_imgs}
    train_list, val_list, skipped = [], [], 0
    mapping = {}
    for p, h in zip(ours, ho):
        if h is None: continue
        is_official = h in official
        if p.stem in val_stems:
            val_list.append(p)                      # val unchanged, for comparability
        elif is_official:
            train_list.append(p)                    # official AND not in our val
            mapping[hexmap[h]] = p.name
        else:
            skipped += 1                            # ours-only -> excluded from training

    (OUT/'official_train.txt').write_text('\n'.join(str(p) for p in train_list) + '\n')
    (OUT/'official_val.txt').write_text('\n'.join(str(p) for p in val_list) + '\n')
    val_official = sum(1 for p in val_list if ho[ours.index(p)] in official) if False else None
    stats = {
        'official_images': len(official),
        'our_images': len(ours),
        'dfine_train': len(train_list),
        'dfine_val': len(val_list),
        'ours_only_excluded_from_training': skipped,
    }
    (OUT/'membership.json').write_text(json.dumps({'stats': stats, 'hex_to_ours': mapping}, indent=2))
    for k, v in stats.items(): print(f'  {k:38} {v}')
    print(f'\nwrote {OUT}/official_train.txt and official_val.txt')

if __name__ == '__main__':
    main()
