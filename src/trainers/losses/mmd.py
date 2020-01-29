import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFMMDLoss(nn.Module):

    name = 'rbf_mmd_loss'

    def __init__(self, sigma=1):
        super(RBFMMDLoss, self).__init__()
        self.sigma = sigma

    def forward(
            self,
            source_features,
            target_features,
        ):
        bs = source_features.size(0)
        bt = target_features.size(0)
        source_features = source_features.view(bs, -1)
        target_features = target_features.view(bt, -1)
        k_xx, k_xy, k_yy = self._rbf_kernel(source_features, target_features)
        mmd2 = self._mmd2(k_xx, k_xy, k_yy)
        return mmd2

    def _rbf_kernel(self, x1, x2):
        batch_size = x1.size(0)
        z = torch.cat((x1, x2), dim=0)
        zzt = torch.mm(z, z.t())
        diag_zzt = torch.diag(zzt).unsqueeze(1)
        z_norm_sqr = diag_zzt.expand_as(zzt)
        exponent = z_norm_sqr - 2 * zzt + z_norm_sqr.t()
        gamma = 1.0 / (2 * self.sigma ** 2)
        k = torch.exp(-gamma * exponent)
        return k[:batch_size, :batch_size], k[:batch_size, batch_size:], k[batch_size:, batch_size:]

    def _mmd2(self, k_xx, k_xy, k_yy):
        m = k_xx.size(0)
        diag_x = torch.diag(k_xx)
        diag_y = torch.diag(k_yy)
        sum_diag_x = torch.sum(diag_x)
        sum_diag_y = torch.sum(diag_y)

        kt_xx_sums = k_xx.sum(dim=1) - diag_x
        kt_yy_sums = k_yy.sum(dim=1) - diag_y
        k_xy_sums_0 = k_xy.sum(dim=0)

        kt_xx_sum = kt_xx_sums.sum()
        kt_yy_sum = kt_yy_sums.sum()
        k_xy_sum = k_xy_sums_0.sum()

        mmd2 = ((kt_xx_sum + sum_diag_x) / (m * m)
                + (kt_yy_sum + sum_diag_y) / (m * m)
                - 2.0 * k_xy_sum / (m * m))

        return mmd2
