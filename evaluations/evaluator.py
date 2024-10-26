import numpy as np
import torch
from explainers.explainer import Explainer

from transformers import AutoTokenizer, BertForSequenceClassification
from explainers.explanation_methods import SHAP
from utils.utils import get_device


class Evaluator():
    """
    Wrapper for evaluation of explanations
    """

    # TODO Discuss what would make sense. maybe init with eval metric?
    def __init__(self, explainer: Explainer):
        self.model = explainer.model   # the 'black box' being explained
        self.tokenizer = explainer.tokenizer
        self.explainer = explainer  # explanation method to evaluate
        self.device = explainer.device

    def evaluate(self, args):
        """
        Method to evaluate a pruning method.
        """
        pass

    def get_local_infidelity(self, input):
        """
        Returns local infidelity of explanation method at input with respect to the model

        params: 
            method: explanation method to evaluate
            input: input being explained

        returns:
            local infidelity
        """
        tokenized_input = self.tokenizer(
            input, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**tokenized_input).logits

        predicted_class_id = logits.argmax().item()
        explanation = self.explainer.explain(input)

        top_feature_id, top_feature = self.get_top_feature(
            explanation, type(self.explainer).__name__, predicted_class_id)

        # perturb input by replacing top feature with PAD token `0`
        tokens = tokenized_input["input_ids"]
        tokens[0, top_feature_id] = 0
        tokenized_input["input_ids"] = tokens

        # get output of model on perturbed input
        with torch.no_grad():
            perturbed_logits = self.model(**tokenized_input).logits

        # calculate local infidelity
        # compares top_feature importance to change in model output after dropping top_feature
        infidelity = (top_feature - (logits[0, predicted_class_id] -
                                     perturbed_logits[0, predicted_class_id]))**2

        return infidelity.item()

    # Gets feature in explanation with greatest contribution to predicted class
    # Returns index and value of top feature
    def get_top_feature(self, explanation, method, predicted_class_id):
        if method == 'SHAP':
            # `explanation` is a 2d array with elts:
            # [contribution to label 0, contribution to label 1]
            top_feature_id = np.argmax(explanation[:, predicted_class_id])
            top_feature = explanation[top_feature_id, predicted_class_id]
        elif method == 'LIME':
            # `explanation` is a list [(token, importance),...]
            # negative feature importance -> feature contributes to label 0
            # positive feature importance -> feature contributes to label 1
            compare = min() if predicted_class_id == 0 else max()
            top_feature_id = compare(
                enumerate(explanation), key=lambda i: i[1][1])[0]
            top_feature = explanation[top_feature_id][1]
        else:
            raise ValueError(f'Explanation method {method} not supported')

        return top_feature_id, top_feature


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")
    model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")
    device = get_device()
    model = model.to(device)

    input = 'Hello, my dog is so terribly ugly'

    explainer = SHAP(model, tokenizer, device)
    evaluator = Evaluator(explainer)
    infidelity = evaluator.get_local_infidelity(input)
    print(infidelity)
