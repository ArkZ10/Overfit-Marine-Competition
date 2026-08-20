"""Trace WHERE the NaN is born for a known-bad sample (default index 229).

Checks, in order: backbone SFP features -> RPN outputs/proposals -> ROI bbox targets.
Prints the first stage that is already non-finite, which is the actual fault site.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 deep_nan_probe.py --index 229
"""
import argparse
import torch
from mmdet.utils import register_all_modules


def stat(name, t):
    if isinstance(t, (list, tuple)):
        for i, x in enumerate(t):
            stat('%s[%d]' % (name, i), x)
        return
    if not torch.is_tensor(t):
        return
    f = torch.isfinite(t)
    print('   %-22s shape %-22s finite %d/%d  min %.4g  max %.4g'
          % (name, tuple(t.shape), int(f.sum()), t.numel(),
             float(t[f].min()) if f.any() else float('nan'),
             float(t[f].max()) if f.any() else float('nan')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/cascade_rcnn_dinov3b_sfp_marine_1024.py')
    ap.add_argument('--index', type=int, default=229)
    args = ap.parse_args()

    register_all_modules()
    import dinov3_sfp  # noqa: F401
    from mmengine.config import Config
    from mmengine.registry import MODELS
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.work_dir = 'runs/_deepprobe'
    cfg.train_dataloader.sampler = dict(type='DefaultSampler', shuffle=False)
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    loader = Runner.build_dataloader(cfg.train_dataloader)

    model = MODELS.build(cfg.model).cuda()
    model.init_weights()
    model.train()

    for i, data in enumerate(loader):
        if i != args.index:
            continue
        data = model.data_preprocessor(data, True)
        imgs = data['inputs']
        samples = data['data_samples']
        print('input image tensor:')
        stat('inputs', imgs)
        print('   padded shape:', tuple(imgs.shape))
        print('   meta img_shape:', samples[0].img_shape,
              ' pad_shape:', getattr(samples[0], 'pad_shape', None))
        gt = samples[0].gt_instances
        print('   gt boxes:', gt.bboxes.cpu().numpy())

        with torch.autocast('cuda', dtype=torch.bfloat16):
            print('\n1) BACKBONE / SFP features:')
            feats = model.extract_feat(imgs)
            stat('feat', feats)

            print('\n2) RPN:')
            rpn_out = model.rpn_head(feats)
            stat('rpn_cls', rpn_out[0])
            stat('rpn_bbox', rpn_out[1])

            proposal_cfg = model.train_cfg.get('rpn_proposal', model.test_cfg.rpn)
            import copy as _c
            rpn_samples = []
            for s in samples:
                s2 = _c.deepcopy(s)
                s2.gt_instances.labels = torch.zeros_like(s2.gt_instances.labels)
                rpn_samples.append(s2)
            losses, rpn_results = model.rpn_head.loss_and_predict(
                feats, rpn_samples, proposal_cfg=proposal_cfg)
            print('   rpn losses:', {k: float(v[0]) if isinstance(v, list) else float(v)
                                     for k, v in losses.items()})
            pb = rpn_results[0].bboxes
            w = pb[:, 2] - pb[:, 0]
            h = pb[:, 3] - pb[:, 1]
            print('   proposals: %d   finite %d/%d' % (len(pb), int(torch.isfinite(pb).sum()), pb.numel()))
            print('   proposal w: min %.6f  h: min %.6f' % (float(w.min()), float(h.min())))
            print('   proposals with w<=0: %d   h<=0: %d' % (int((w <= 0).sum()), int((h <= 0).sum())))

            print('\n3) ROI HEAD loss:')
            roi_losses = model.roi_head.loss(feats, rpn_results, samples)
            for k, v in roi_losses.items():
                print('   %-18s %s' % (k, float(v) if torch.is_tensor(v) else v))
        break


if __name__ == '__main__':
    main()
