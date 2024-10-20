""" 
    Explanation methods implementing `Explainer` abstract base class
"""


import lime.lime_text
import shap
import lime
from torch.nn.functional import softmax
from transformers import pipeline

from explainers.explainer import Explainer


class SHAP(Explainer):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def explain(self, input):
        self.model = pipeline('text-classification',
                              model=self.model,
                              device=0,
                              tokenizer=self.tokenizer,
                              top_k=None)
        explainer = shap.Explainer(self.model)
        shap_values = explainer([input])

        return shap_values


class LIME(Explainer):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def predict_proba(self, texts):
        """
        Prediction function for LIME explainer

        params:
            texts (list of str): List of input texts

        returns:
            predictions (numpy.ndarray): Array of prediction probabilities
        """
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        probabilities = softmax(outputs.logits, dim=-1).detach().numpy()

        return probabilities

    def explain(self, input):
        labels = self.model.config.label2id.keys()
        explainer = lime.lime_text.LimeTextExplainer(class_names=labels)
        exp = explainer.explain_instance(input, self.predict_proba)

        return exp.as_list()
