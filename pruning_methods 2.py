import torch.nn.utils.prune as prune
from typing import List, Tuple
import torch
import torch.nn as nn

class Pruner():
    """
    Wrapper around pruning methods, allowing easy access
    """
    def __init__(self, model):
        self.model = model

    def prune(self, method):
        """
        Prune a model with a given method

        Params:
            model: Model to be pruned.
            method: Pruning method to use. Should be instance of PruningMethod class.
        """
        # Randomly prunes user defined percentage of the weights in paras_to_prune
        if method.type == "RandomUnstructured":
            prune.global_unstructured(method.paras_to_prune, pruning_method=prune.RandomUnstructured, amount=method.percentage)
            #method.remover()

        # Prunes the lowest n percent of the weights in paras_to_prune ranked by absolute value (L1-norm)
        # n is defined by the percentage attribute
        elif method.type == "L1Unstructured":
            prune.global_unstructured(method.paras_to_prune, pruning_method=prune.L1Unstructured, amount=method.percentage)
            #method.remover()

        # Custom user defined pruning mask
        # Prunes the weights in paras_to_prune according to the mask
        elif method.type == "Custom":
            for module, param_name in method.paras_to_prune:
                prune.custom_from_mask(module, name=param_name, mask=method.mask)
            #method.remover()
        else: 
            print("Pruning method unknown. Supported pruning methods are: RandomUnstructured, L1Unstructured and Custom.")




class PruningMethod():
    """
    Wrapper for pruning methods, allowing to store all necessary attributes in method instance
    """
    def __init__(self, type: str, 
                 paras_to_prune: List[Tuple[nn.Module, str]], 
                 percentage: float, 
                 mask: torch.Tensor):
        
        # type: Pruning method to use. Should be one of the following: RandomUnstructured, L1Unstructured, Custom
        self.type = type
        # paras_to_prune: List of tuples containing the module and parameter name to be pruned
        self.paras_to_prune = paras_to_prune
        # percentage: Percentage of weights to be pruned
        self.percentage = percentage
        # mask: Pruning mask to be used for custom pruning
        self.mask = mask

    # TODO: discuss necessity.
    def remover(self):
        # Remove the pruning mask from each pruned parameter
        for (module, name) in self.paras_to_prune:
            prune.remove(module, name)

        """Discuss if we want to remove before or after fine tuning"""


