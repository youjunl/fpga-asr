from turtle import forward
import torch
import torch.nn as nn


class CTCLossNM(nn.Module):
    """
    Neural Module wrapper for pytorch's ctcloss

    Args:
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
    """

    def __init__(self, num_classes, **kwargs):
        nn.Module.__init__(self)

        # self._blank = self.local_parameters.get('blank', 0)
        self._blank = num_classes
        self._criterion = nn.CTCLoss(blank=self._blank,
                                     reduction='none')

    def _loss(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets,
                               input_length,
                               target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        loss = torch.mean(loss)
        return loss

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))
