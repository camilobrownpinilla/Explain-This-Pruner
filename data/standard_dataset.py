from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import PreTrainedTokenizer


class StandardDataset(ABC):
    """Base class from which datasets will inherit. Standardizes splits and 
       mapping
    """

    @abstractmethod
    def train(self) -> Dataset:
        """
        Return the training set of the dataset.
        """
        pass
    
    @abstractmethod
    def test(self) -> Dataset:
        """
        Return the test set of the dataset.
        """
        pass
    
    @abstractmethod
    def encode(self, tokenizer: PreTrainedTokenizer):
        """
        Returns a function that, given a raw example, encodes said example using
        'tokenizer'.

        Args:
            tokenizer: HF compatible tokenizer.
        """
