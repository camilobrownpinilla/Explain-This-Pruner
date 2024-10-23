"""
    Pruning methods implementing `Pruner` abstract base class
"""

import torch.nn.utils.prune as prune
import torch

from pruners.pruner import Pruner


class RandUnstructured(Pruner):
    def __init__(self, model, params, ptg):  
        super().__init__(model, params, ptg)
        self.method = prune.RandomUnstructured

    def __call__(self):    
        # Randomly prunes `ptg` percentage of `params`
        prune.global_unstructured(
            self.params, pruning_method=self.method, amount=self.ptg)
        super().remove()


class L1Unstructured(Pruner):
    def __init__(self, model, params, ptg):
        super().__init__(model, params, ptg)
        self.method = prune.L1Unstructured

    def __call__(self):
        # Prunes the lowest `ptg` percent of `params` ranked by absolute value
        prune.global_unstructured(
            self.params, pruning_method=self.method, amount=self.ptg)
        super().remove()


class CustomMask(Pruner):
    def __init__(self, model, params, ptg, mask: torch.Tensor):
        super().__init__(model, params, ptg)
        self.mask = mask

    def __call__(self):
        # Prunes the weights in `params` according to the mask
        for module, param_name in self.params:
            prune.custom_from_mask(module, name=param_name, mask=self.mask)
        super().remove()
