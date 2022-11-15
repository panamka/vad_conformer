import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self, treshold=0.5):
        super().__init__()
        self.treshold = treshold

    def forward(self, pred_label, gt_label):
        pred_label = pred_label.unsqueeze(1)
        gt_label = gt_label.unsqueeze(1)
        pred_label = pred_label > self.treshold

        result = torch.eq(gt_label, pred_label).float().mean((1,2))
        return result

class MseLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred_label, gt_label):
        pred_label = pred_label.unsqueeze(1)
        gt_label = gt_label.unsqueeze(1)

        pred_label = self.sigmoid(pred_label)
        result = self.criterion(pred_label, gt_label).mean((1, 2))
        return result


