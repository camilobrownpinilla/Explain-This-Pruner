from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from data.standard_dataset import StandardDataset


__all__ = ["IMDB", "YelpPolarity", "Emotion"]


class IMDB(StandardDataset):
    x = 'review'  # name of col containing text samples
    
    def __init__(self):
        self.dataset = load_dataset("ajaykarthick/imdb-movie-reviews")

    def train(self):
        return self.dataset['train']
    
    def test(self):
        return self.dataset['test']
    
    @staticmethod
    def encode(tokenizer):
        return lambda example: tokenizer(example[IMDB.x], truncation=True)
    

class YelpPolarity(StandardDataset):
    x = 'text'
    
    def __init__(self):
        self.dataset = load_dataset("fancyzhx/yelp_polarity")

    def train(self):
        return self.dataset['train']
    
    def test(self):
        return self.dataset['test']
    
    @staticmethod
    def encode(tokenizer):
        return lambda example: tokenizer(example[YelpPolarity.x], truncation=True)
    

class Emotion(StandardDataset):
    x = 'text'
    
    def __init__(self):
        self.dataset = load_dataset("dair-ai/emotion")

    def train(self):
        return self.dataset['train']
    
    def test(self):
        return concatenate_datasets([self.dataset['test'], self.dataset['validation']])
    
    @staticmethod
    def encode(tokenizer):
        return lambda example: tokenizer(example[Emotion.x], truncation=True)
    

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    imdb = Emotion()
    train, test = imdb.train(), imdb.test()
    imdb.dataset.map(imdb.encode(tokenizer))
