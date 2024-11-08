"""
    Pruning methods implementing `Pruner` abstract base class
"""

import torch.nn.utils.prune as prune
import torch

from nn_pruning import MovementPruningConfig, ModelPatcher
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from torchvision.models import resnet18
import torch_pruning as tp



from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter

from pruners.pruner import Pruner

__all__ =  ['RandUnstructured',' L1Unstructured', 'CustomMask', 'OptimalBrainSurgeon']

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






# Second  order pruning method
# SparseML's OptimalBrainSurgeon
# This pruning methods should be used while finetuning the model (possible while training) not as a standalone method
# Should be used in conjunction with a training loop
class OptimalBrainSurgeon(Pruner):
    def __init__(self):
        pass
    # model should be huggingface model, pruning_schedule should be a yaml filename string
    def prune(self, model, pruning_schedule):
        manager = ScheduledModifierManager.from_yaml(pruning_schedule) # "pruning_schedule.yaml"
        manager.initialize(model)  

        for epoch in range(num_epochs):
            manager.modify(model, epoch=epoch)  # Apply the pruning schedule



# First order pruning method
class MovementPruning(Pruner):
    def __init__(self, model, pruning_config):
        self.model = model
        self.pruning_config = MovementPruningConfig(**pruning_config)
        self.patcher = ModelPatcher(self.model, self.pruning_config)

    def prune(self, params):
        # Apply the pruning
        self.patcher.patch()

    


# Dep Graph Pruning Interface
class DepGraphPruning(Pruner):
    def __init__(self, model, pruning_config):
       pass

    def prune(self, params):
        pass