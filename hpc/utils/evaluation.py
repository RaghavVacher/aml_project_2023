import torch
import torch.nn as nn

class MNNPC(nn. Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, nc=None):
        pred_mean = torch.mean(pred, axis=0)
        gt_mean = torch.mean(gt, axis=0)

        ts = gt - gt_mean
        pred_t = pred - pred_mean

        msi = (gt - gt_mean).view(-1, 1) * (pred_t - pred_mean.view(-1, 1))
        ms1 = (gt - gt_mean).view(-1, 1) ** 2
        ms2 = (pred_t - pred_mean.view(-1, 1)) ** 2
        ms = (msi * ms2) ** 0.5

        rv = ts.sum(axis=1) / (ms + 1e-8)
        rv = (rv + 1) / 2

        if nc is not None:
            nc[nc == 0] = 1
            rv = rv ** 2 / torch.from_numpy(nc).to(rv.device)

        return rv.mean()