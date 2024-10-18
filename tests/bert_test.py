from transformers import AutoTokenizer, BertForSequenceClassification
from explainers.explainer import Explainer

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

explainer = Explainer(model)

explainer.explain('shap', inputs)

