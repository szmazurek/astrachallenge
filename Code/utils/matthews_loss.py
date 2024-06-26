import torch
import torch.nn as nn

class MCC_Loss(nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self, zero_division=1e-6):
        super(MCC_Loss, self).__init__()
        self.zero = zero_division

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp, alpha=1)
            * torch.add(tp, fn, alpha=1)
            * torch.add(tn, fp, alpha=1)
            * torch.add(tn, fn, alpha=1)
        )

        # Adding epsilon to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + self.zero)
        return 1 - mcc