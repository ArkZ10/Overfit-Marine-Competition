"""Attribute the candidate-Z NaN to a specific training sample.

Both runs went NaN after exactly 1400 IMAGES (batch1 iter1400, batch4 iter350) under the
same seed, so this is a poisoned sample, not LR divergence. This walks the train set in
index order, runs a real forward, and reports every sample whose loss is non-finite --
plus enough state to say WHY.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 find_nan_sample.py --limit 2000
"""
import argparse

import torch

from mmdet.utils import register_all_modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/cascade_rcnn_dinov3b_sfp_marine_1024.py')
    ap.add_argument('--limit', type=int, default=2000)
    args = ap.parse_args()

    register_all_modules()
    import dinov3_sfp  # noqa: F401  (registers the backbone)
    from mmengine.config import Config
    from mmengine.registry import MODELS
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.work_dir = 'runs/_nanprobe'

    # deterministic order, no shuffling: we want index-attributable results
    cfg.train_dataloader.sampler = dict(type='DefaultSampler', shuffle=False)
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 4
    loader = Runner.build_dataloader(cfg.train_dataloader)

    model = MODELS.build(cfg.model).cuda()
    model.init_weights()
    model.train()

    ds = loader.dataset
    hits = 0
    for i, data in enumerate(loader):
        if i >= args.limit:
            break
        data = model.data_preprocessor(data, True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = model.loss(data['inputs'], data['data_samples'])
        flat = {}
        for k, v in losses.items():
            if isinstance(v, (list, tuple)):
                for j, vv in enumerate(v):
                    flat['%s[%d]' % (k, j)] = float(vv)
            else:
                flat[k] = float(v)
        bad = {k: v for k, v in flat.items() if v != v or abs(v) == float('inf')}
        if bad:
            hits += 1
            info = ds.get_data_info(i)
            gts = info.get('instances', [])
            ws = [g['bbox'][2] - g['bbox'][0] for g in gts]
            hs = [g['bbox'][3] - g['bbox'][1] for g in gts]
            print('')
            print('### NON-FINITE at index %d' % i)
            print('  file      : %s' % str(info.get('img_path', '?')).split('/')[-1])
            print('  img size  : %sx%s' % (info.get('width'), info.get('height')))
            print('  n_gt      : %d' % len(gts))
            if ws:
                print('  gt w range: %.2f .. %.2f' % (min(ws), max(ws)))
                print('  gt h range: %.2f .. %.2f' % (min(hs), max(hs)))
            print('  bad losses: %s' % bad)
            if hits >= 5:
                print('')
                print('(stopping after 5 hits)')
                break
        if i % 200 == 0:
            print('  ...%d scanned, %d non-finite' % (i, hits), flush=True)
    print('')
    print('scanned %d samples, %d non-finite' % (min(args.limit, len(ds)), hits))


if __name__ == '__main__':
    main()
