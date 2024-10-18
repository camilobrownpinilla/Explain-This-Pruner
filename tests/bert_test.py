from transformers import AutoTokenizer, BertForSequenceClassification
from explainers.explainer import Explainer

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

input = 'Hello, my dog is cute'
tokenized_inputs = tokenizer(input)

explainer = Explainer(model, tokenizer)

explanation = explainer.explain('lime', input)

print(explanation)


