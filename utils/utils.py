import torch
import torch.nn as nn


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_params_to_prune(model):
    """Returns all non-embedding parameters of a PyTorch model for pruning.

    Args:
        model: Model whose parameters are to be pruned

    Returns:
        List: List of parameters to be passed to a PyTorch pruning method
    """
    model_params = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Embedding):  # Skip embedding layers
            if hasattr(module, 'weight'):  # Ensure it has a weight parameter
                model_params.append((module, 'weight'))

    return model_params

def reinitialize_weights(module):
    """Reinitializes weights of a model"""
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    else:
        for param in module.parameters():
            if param.dim() > 1:  # usually for weight matrices
                torch.nn.init.xavier_uniform_(param)
            else:  # usually for biases or 1D vectors
                torch.nn.init.zeros_(param)
