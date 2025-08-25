from functools import partial, reduce
import pandas as pd
import numpy as np 
import os
import sys
from torch import nn
import torch.nn.functional as F
import torch
import h5py
from pathlib import Path
from functools import partial
from loss_modules import _WeightedLoss, _Loss
## distance funcs

def sigmoid_cosine_distance_p(x, y, p=1): # p is weighting factor
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p

DISTANCE_FN_DICT = {'sigmoid_cosine_distance_p': sigmoid_cosine_distance_p,
                   'euclidean': None}

def tanh_decay():
     
    return

def cosine_decay():
    
    return

class MarginScheduledTripletLossFunction(nn.Module):
    def __init__(self, p, distance, margin=0.25, seed=123):
        super().__init__()
        ## could change into sigmoid_cosine_distance_p
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function= distance ,margin=margin, swap=False, reduction='mean')
        self.seed = seed
    
    def forward(self, latent, pos_len):
        """
        Args:
            latent: [total_residues, embedding_dim] - embeddings from model
            labels: [total_residues] - 1 for positive, 0 for negative  
            pos_len: list of positive counts per protein in batch
        """
        torch.manual_seed(self.seed)
        
        pos_idx = torch.arange(0, pos_len.item())
        neg_idx = torch.arange(pos_len.item(), latent.shape[0])
        
        num_triplets = len(pos_idx)
        pos_samples = pos_idx[torch.randint(0, len(pos_idx), (num_triplets,))]
        neg_samples = neg_idx[torch.randint(0, len(neg_idx), (num_triplets,))]
        
        ## slice 
        anchors = latent[pos_idx]
        positives = latent[pos_samples]
        negatives = latent[neg_samples]
        
        loss = self.criterion(anchors, positives, negatives)
        
        return loss  # Avoid division by zero


# # distance
# def sigmoid_cosine_distance_p(x, y, p=1.0, dim=-1, tau=1.0, eps=1e-8):
#     sim = F.cosine_similarity(x, y, dim=dim, eps=eps)   # [-1, 1]
#     d = 1.0 - torch.sigmoid(sim / tau)                  # ~[0,1] as tau→0
#     return d.pow(p)

# def tanh_decay(M_0, N_epoch, x):
#     return float(M_0 * (1.0 - np.tanh(2.0 * x / N_epoch)))  # ~M0→~0.036*M0

# def cosine_anneal(M_0, N_epoch, x):
#     return float(0.5 * M_0 * (1.0 + np.cos(np.pi * x / N_epoch)))  # M0→0

# def no_decay(M_0, N_epoch, x):
#     return float(M_0)

# MARGIN_FN_DICT = {
#     "tanh_decay": tanh_decay,
#     "cosine_anneal": cosine_anneal,
#     "no_decay": no_decay,
# }

# class MarginScheduledLossFunction:
#     def __init__(self, M_0: float = 0.25, N_epoch: int = 50, N_restart: int = -1,
#                  update_fn: str = "tanh_decay", p: float = 1.0, tau: float = 1.0, dim: int = -1):
#         if update_fn not in MARGIN_FN_DICT:
#             raise ValueError(f"Unknown update_fn: {update_fn}. Choose from {list(MARGIN_FN_DICT)}")
#         if N_epoch <= 0:
#             raise ValueError("N_epoch must be > 0")
#         if N_restart == -1:
#             N_restart = N_epoch
#         if N_restart <= 0:
#             raise ValueError("N_restart must be > 0")

#         self.M_0 = float(M_0)
#         self.N_epoch = int(N_epoch)
#         self.N_restart = int(N_restart)
#         self._step = 0

#         # schedule
#         self._update_margin_fn = partial(MARGIN_FN_DICT[update_fn], self.M_0, self.N_restart)
#         self.M_curr = self._update_margin_fn(0)

#         # distance params
#         self._dist = partial(sigmoid_cosine_distance_p, p=p, dim=dim, tau=tau)

#     @property
#     def margin(self) -> float:
#         return self.M_curr

#     def step(self):
#         # modulo-progress within the current cycle (includes x=0 and x=N_restart)
#         self._step += 1
#         x = self._step % self.N_restart
#         self.M_curr = self._update_margin_fn(x)

#     def reset(self):
#         self._step = 0
#         self.M_curr = self._update_margin_fn(0)

#     def __call__(self, anchor, positive, negative):
#         # functional variant so we can pass a dynamic margin
#         return F.triplet_margin_with_distance_loss(
#             anchor, positive, negative,
#             distance_function=self._dist,
#             margin=float(self.M_curr),
#             swap=False,
#             reduction='mean'
#         )


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
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )