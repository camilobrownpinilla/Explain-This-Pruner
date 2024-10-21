"""
    Abstract Pruner base class
    Implemented by classes in `pruning_methods.py`
"""

import torch.nn.utils.prune as prune

from abc import ABC, abstractmethod


class Pruner(ABC):
    """Wrapper around model for pruning its parameters"""

    def __init__(self, model, params, ptg):
        """
        Args:
            model: model to prune
            params: model parameters to prune
            ptg (float): percentage of params to be pruned
        """
        self.model = model
        self.params = params
        self.ptg = ptg

    # Applies pruning mask to the model
    @abstractmethod
    def __call__(self):
        pass

    # Removes the parameters with pruning mask
    def remove(self):
        for (module, name) in self.params:
            prune.remove(module, name)
