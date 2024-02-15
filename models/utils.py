from torch.nn.modules.loss import _Loss
import torch

class CrossEntropyVecLoss(_Loss):
    def forward(self, out, gt):
        bs = out.size(0)
        loss = - torch.mul(gt.float(), torch.log(out.float() + 1e-7))
        loss = loss.sum(dim = 1)
        return loss


##### model init #####
def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        if hasattr(m, "weight"):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif type(m) == torch.nn.Linear:
        if hasattr(m, "weight"):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                