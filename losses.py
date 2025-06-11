import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred_flat.sum() + target_flat.sum() + smooth))