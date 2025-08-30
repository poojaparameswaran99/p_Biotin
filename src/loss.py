from functools import partial, reduce
import pandas as pd
import numpy as np 
import os
import sys
from torch import nn
import torch.nn.functional as F
import torch
import h5py
from typing import Optional
from torch import Tensor
from pathlib import Path
from functools import partial
from torch.nn.modules.loss import _WeightedLoss, _Loss
## distance funcs

def sigmoid_cosine_distance_p(x, y, p=1): # p is weighting factor
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p

DISTANCE_FN_DICT = {'sigmoid_cosine_distance_p': sigmoid_cosine_distance_p,
                   'euclidean': None}

def tanh_decay(M_0, N_restart, x):
    return float(M_0 * (1.0 - np.tanh(2.0 * x / max(1, N_restart))))

def cosine_anneal(M_0, N_restart, x):
    return float(0.5 * M_0 * (1.0 + np.cos(np.pi * x / max(1, N_restart))))

def no_decay(M_0, N_restart, x):
    return float(M_0)

MARGIN_FN_DICT = {
    "tanh_decay": tanh_decay,
    "cosine_anneal": cosine_anneal,
    "no_decay": no_decay,
}


class MarginScheduledTripletLossFunction():
    def __init__(self, distance_fn, anneal_fn, N_restart: int =10, seed=123, step_per_call=False):
        ## could change into sigmoid_cosine_distance_p
        self._dist = distance_fn        
        self._anneal = anneal_fn  
        self.N_restart = int(N_restart)
        self._step = 0
        self.step_per_call = bool(step_per_call)
        self.M_curr = float(self._anneal(x=0))
        self.seed = seed
    
    ## only referenced upon call on local script
    @property
    def margin(self) -> float:
        return float(self.M_curr)

    def step(self):
        self._step += 1
        x = self._step % self.N_restart
        self.M_curr = float(self._anneal(x=x)) ## update margin fn

    def reset(self):
        self._step = 0
        self.M_curr = self._anneal(x=0) ## update_margin_fn

    @torch.no_grad()
    def _sample_indices(self, N_pos, N_total, device):
        # anchors: all positives [0..N_pos-1]
        pos_idx = torch.arange(N_pos, device=device)
        print(pos_idx)
        neg_idx = torch.arange(N_pos, N_total, device=device)

        if neg_idx.numel() == 0:
            raise ValueError("No negatives available: N_total must be > N_pos.")
        # pos could be self
        pos_perm = torch.randperm(N_pos, device=device)
        pos_samples = pos_idx[pos_perm]
        neg_samples = neg_idx[torch.randperm(len(neg_idx), device=device)[:N_pos+1]]
        print('in loss, pos samples len: ', len(pos_samples))
        print('in loss, neg samples len: ', len(neg_samples))
        
        return pos_idx, pos_samples, neg_samples

    def __call__(self, latent: torch.Tensor, pos_len):
        if isinstance(pos_len, torch.Tensor):
            pos_len = int(pos_len.item())
        N, D = latent.shape
        if not (0 < pos_len < N):
            raise ValueError(f"pos_len must be in (0,{N}), got {pos_len}.")

        device = latent.device
        pos_idx, pos_samples, neg_samples = self._sample_indices(pos_len, N, device)

        anchors   = latent[pos_idx]       # [pos_len, D]
        positives = latent[pos_samples]   # [pos_len, D]
        negatives = latent[neg_samples]   # [pos_len, D]
        loss = F.triplet_margin_with_distance_loss(
            anchors, positives, negatives,
            distance_function=self._dist,
            margin=self.margin,
            swap=False,
            reduction='mean'
        )

        if self.step_per_call:
            self.step()

        return loss



class BCELoss(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy between the target and
    the input probabilities:

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets :math:`y` should be numbers
    between 0 and 1.

    Notice that if :math:`x_n` is either 0 or 1, one of the log terms would be
    mathematically undefined in the above loss equation. PyTorch chooses to set
    :math:`\log (0) = -\infty`, since :math:`\lim_{x\to 0} \log (x) = -\infty`.
    However, an infinite term in the loss equation is not desirable for several reasons.

    For one, if either :math:`y_n = 0` or :math:`(1 - y_n) = 0`, then we would be
    multiplying 0 with infinity. Secondly, if we have an infinite loss value, then
    we would also have an infinite term in our gradient, since
    :math:`\lim_{x\to 0} \frac{d}{dx} \log (x) = \infty`.
    This would make BCELoss's backward method nonlinear with respect to :math:`x_n`,
    and using it for things like linear regression would not be straight-forward.

    Our solution is that BCELoss clamps its log function outputs to be greater than
    or equal to -100. This way, we can always have a finite loss value and a linear
    backward method.


    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

    Examples:

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, 2, requires_grad=True)
        >>> target = torch.rand(3, 2, requires_grad=False)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean"
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )