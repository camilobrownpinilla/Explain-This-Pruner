from transformers import AutoTokenizer, BertForSequenceClassification
import torch

from explainers.explanation_methods import LIME


tokenizer = AutoTokenizer.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity")
print('model loaded')

input = 'My dog is cute and evil'
tokenized_input = tokenizer(input, return_tensors="pt")

explainer = LIME(model, tokenizer)
explanation = explainer.explain(input)
print('LIME explanation: ', explanation)

with torch.no_grad():
    logits = model(**tokenized_input).logits

predicted_class_id = logits.argmax().item()
predicted_class = model.config.id2label[predicted_class_id]
print(f'predicted class id: {predicted_class_id}')
print(f'predicted class: {predicted_class}')
