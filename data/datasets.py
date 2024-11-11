from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from data.standard_dataset import StandardDataset


__all__ = ["IMDB", "YelpPolarity", "Emotion"]


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
    imdb = YelpPolarity()
    train, test = imdb.train(), imdb.test()
    imdb.dataset.map(imdb.encode(tokenizer))
