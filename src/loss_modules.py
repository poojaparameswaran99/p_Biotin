import torch
from torch.nn import Module

class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
            
            

class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.weight: Optional[Tensor]
