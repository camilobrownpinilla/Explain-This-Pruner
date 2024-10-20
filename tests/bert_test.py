from transformers import AutoTokenizer, BertForSequenceClassification
from explainers.explanation_methods import LIME


tokenizer = AutoTokenizer.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity")
print('model loaded')

input = 'Hello, my dog is cute'
tokenized_inputs = tokenizer(input)

explainer = LIME(model, tokenizer)
explanation = explainer.explain(input)
print('LIME explanation: ', explanation)
