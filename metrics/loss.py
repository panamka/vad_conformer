import torch.nn as nn
import torch


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_label, gt_label):
        pred_label = pred_label.unsqueeze(1)
        gt_label = gt_label.unsqueeze(1)
        result = self.criterion(pred_label, gt_label)
        result = result.mean((1, 2))
        return result


def main():
    loss = CrossEntropy()

    input = torch.randn(3, 256, 256)
    target = torch.randn(3, 256, 256)
    output = loss(input, target)
    print(output.shape)

if __name__ == '__main__':
    main()


