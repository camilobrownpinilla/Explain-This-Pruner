"""
    Abstract Explainer base class
    Defines abstract methods that must be implemented
    by children classes (e.g. SHAP, LIME)
"""

from transformers.modeling_utils import PreTrainedModel
from abc import ABC, abstractmethod


class Explainer(ABC):
    """Wrapper around model for explaining outputs"""

    def __init__(self, model, tokenizer, device):
        assert isinstance(model, PreTrainedModel), \
            "Currently only HF transformers are supported"
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def explain(self, input: str):
        """
        Explains model's output for `input`

        params:
            input (str): input to be passed to model

        returns:
            explanation of model's output
        """
        pass
