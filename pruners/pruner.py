"""
    Abstract Pruner base class
    Implemented by classes in `pruning_methods.py`
"""

import torch.nn.utils.prune as prune

from abc import ABC, abstractmethod


class Pruner(ABC):
    """Base class from which subclasses implement pruning methods"""

    def __init__(self):
        pass

    # Applies pruning mask to the model
    @abstractmethod
    def prune(self, params, ptg):
        """
        Args:
            params: model parameters to prune
            ptg (float): percentage of params to be pruned
        """
        pass

    # Removes the parameters with pruning mask
    def remove(self, params):
        # params: model parameters to prune
        for (module, name) in params:
            prune.remove(module, name)
