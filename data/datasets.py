from abc import ABC, abstractmethod
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer, AutoTokenizer

__all__ = ["IMDB", "YelpPolarity", "Emotion"]


class StandardDataset(ABC, Dataset):
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


class IMDB(StandardDataset):
    def __init__(self):
        self.dataset = load_dataset("ajaykarthick/imdb-movie-reviews")

    def train(self):
        return self.dataset['train']
    
    def test(self):
        return self.dataset['test']
    
    @staticmethod
    def encode(tokenizer):
        return lambda example: tokenizer(example['review'], truncation=True)
    

class YelpPolarity(StandardDataset):
    def __init__(self):
        self.dataset = load_dataset("fancyzhx/yelp_polarity")

    def train(self):
        return self.dataset['train']
    
    def test(self):
        return self.dataset['test']
    
    @staticmethod
    def encode(tokenizer):
        return lambda example: tokenizer(example['text'], truncation=True)
    

class Emotion(StandardDataset):
    def __init__(self):
        self.dataset = load_dataset("dair-ai/emotion")

    def train(self):
        return self.dataset['train']
    
    def test(self):
        return concatenate_datasets([self.dataset['test'], self.dataset['validation']])
    
    @staticmethod
    def encode(tokenizer):
        return lambda example: tokenizer(example['text'], truncation=True)
    


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    imdb = IMDB()
    train, test = imdb.train(), imdb.test()
    imdb.dataset.map(imdb.encode(tokenizer))
