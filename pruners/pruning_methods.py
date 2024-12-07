"""
    Pruning methods implementing `Pruner` abstract base class
"""

import torch.nn.utils.prune as prune
import torch

from pruners.pruner import Pruner

__all__ =  ['RandUnstructured',' L1Unstructured', 'CustomMask', 'L1Structured']

class RandUnstructured(Pruner):
    def __init__(self):
        self.method = prune.RandomUnstructured

    def prune(self, params, ptg):
        # Randomly prunes `ptg` percentage of `params`
        prune.global_unstructured(
            params, pruning_method=self.method, amount=ptg)
        super().remove(params)


class L1Unstructured(Pruner):
    def __init__(self):
        self.method = prune.L1Unstructured

    def prune(self, params, ptg):
        # Prunes the lowest `ptg` percent of `params` ranked by absolute value
        prune.global_unstructured(
            params, pruning_method=self.method, amount=ptg)
        super().remove(params)


class CustomMask(Pruner):
    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def prune(self, params):
        # Prunes the weights in `params` according to the mask
        for module, param_name in params:
            prune.custom_from_mask(module, name=param_name, mask=self.mask)
        super().remove(params)


class LnStructured(Pruner):
    def __init__(self, n = 2):
        self.n = n

    def prune(self, params, ptg):
        for module, name in params:
            prune.ln_structured(module, name, ptg, self.n, dim=1)
        super().remove(params)
