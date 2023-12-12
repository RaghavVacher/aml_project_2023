import torch
import torch.nn as nn

class MNNPC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, nc=None):
        pred_mean = torch.mean(pred, axis=0)
        gt_mean = torch.mean(gt, axis=0)
        gt_t = gt.T
        pred_t = pred.T

        ts = (gt_t - gt_mean.view(-1, 1)) * (pred_t - pred_mean.view(-1, 1))
        ts = ts.sum(axis=1)

        ms1 = (gt_t - gt_mean.view(-1, 1)) ** 2
        ms1 = ms1.sum(axis=1)

        ms2 = (pred_t - pred_mean.view(-1, 1)) ** 2
        ms2 = ms2.sum(axis=1)
        ms = (ms1 * ms2) ** 0.5

        rv = ts / (ms + 1e-8)
        rv = 1 - (rv + 1) / 2

        if nc is not None:
            nc[nc == 0] = 1
            rv = rv**2 / torch.from_numpy(nc).to(rv.device)

        return rv.mean()