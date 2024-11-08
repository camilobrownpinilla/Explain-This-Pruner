import numpy as np
import torch
from explainers.explainer import Explainer

from transformers import AutoTokenizer, BertForSequenceClassification
from explainers.explanation_methods import SHAP, IG
from utils.utils import get_device


class Evaluator():
    """
    Wrapper for evaluation of explanations
    """
    MASK = 103  # [MASK] token maps to 103

    # TODO Discuss what would make sense. maybe init with eval metric?
    def __init__(self, explainer: Explainer):
        self.model = explainer.model   # the 'black box' being explained
        self.tokenizer = explainer.tokenizer
        self.explainer = explainer  # explanation method to evaluate
        self.device = explainer.device

    def evaluate_infidelity_mask_top_k(self, test_set, k=1, num_samples=None):
        """
        Returns the average local infidelity of the explanation method over `num_samples` rand samples in `test_set`.
        Computes local infidelity by masking top k features of an input sample.

        params:
            test_set: a `Dataset` object, e.g. obtained via `load_dataset("imdb")["test"]`
            k: number of top features to mask
            num_samples: number of test samples to evaluate on

        returns:
            average infidelity
        """
        # evaluate on full test set by default
        if num_samples is None:
            num_samples = len(test_set)

        # shuffle test set for random sampling
        shuffled_set = test_set.shuffle(seed=4)  # set seed for reproducibility
        infid = 0
        for sample in shuffled_set[:num_samples]:
            infid += self.get_local_infidelity_mask_top_k(sample["text"], k)

        return infid / num_samples

    def get_local_infidelity_mask_top_k(self, input, k):
        """
        Returns local infidelity of explanation method at input with respect to the model
        Input is perturbed by masking top k features

        params: 
            input: input being explained
            k: number of top features to mask

        returns:
            local infidelity
        """
        tokenized_input = self.tokenizer(
            input, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**tokenized_input).logits

        predicted_class_id = logits.argmax().item()
        explanation = self.explainer.explain(input)

        top_feature_ids, top_features = self.get_top_k_features(
            explanation, type(self.explainer).__name__, predicted_class_id, k)

        # perturb input by replacing top k features with MASK token `103`
        tokens = tokenized_input["input_ids"]
        tokens[0, top_feature_ids] = self.MASK
        tokenized_input["input_ids"] = tokens

        # get output of model on perturbed input
        with torch.no_grad():
            perturbed_logits = self.model(**tokenized_input).logits

        # calculate local infidelity
        # compares top_feature importance to change in model output after dropping top_feature
        infidelity = (sum(top_features) - (logits[0, predicted_class_id] -
                                           perturbed_logits[0, predicted_class_id]))**2

        return infidelity.item()

    def get_top_k_features(self, explanation, method, predicted_class_id, k=1):
        """
        Gets top k features with greatest contribution to predicted class

        params:
            explanation: a feature-attribution explanation of some sample
            method: explanation method used to produce `explanation`
            predicted_class_id: index of predicted class for sample being explained
            k: number of top features to get

        returns:
            top_feature_ids: list of indices of top k features
            top_features: list of top k feature importances
        """
        if k >= len(explanation) or k < 1:
            raise ValueError(
                'k must be at least 1 and no greater than the number of features')

        if method == 'SHAP':
            # `explanation` is a 2d array with elts:
            # [contribution to label 0, contribution to label 1]
            top_feature_ids = sorted(
                range(len(explanation)), key=lambda i: explanation[i][predicted_class_id], reverse=True)[:k]
            top_features = explanation[top_feature_ids, predicted_class_id]

        elif method == 'LIME':
            # `explanation` is a list [(token, importance),...]
            # negative feature importance -> feature contributes to label 0
            # positive feature importance -> feature contributes to label 1
            if predicted_class_id == 0:
                # Get the indices of the k smallest importance values (contributing to label 0)
                top_feature_ids = sorted(
                    range(len(explanation)), key=lambda i: explanation[i][1])[:k]
            else:
                # Get the indices of the k largest importance values (contributing to label 1)
                top_feature_ids = sorted(
                    range(len(explanation)), key=lambda i: explanation[i][1], reverse=True)[:k]

            top_features = [explanation[i][1] for i in top_feature_ids]

        elif method == 'IG':
            # `explanation` is a list of feature importance scores w.r.t. model prediction
            top_feature_ids = sorted(
                range(len(explanation)), key=lambda i: explanation[i], reverse=True)[:k]
            top_features = [explanation[i] for i in top_feature_ids]

        else:
            raise ValueError(f'Explanation method {method} not supported')

        return top_feature_ids, top_features

    def get_local_infidelity_mask_pairs(self, input):
        """
        Returns local infidelity of explanation method at input with respect to the model
        Input is perturbed by masking pairs of features, one pair at a time

        params: 
            input: input being explained

        returns:
            local infidelity
        """
        pass


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")
    model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")
    device = torch.device('cpu')
    model = model.to(device)

    input1 = 'Hello, my dog is so terribly ugly'
    input2 = 'I am very happy about this restaurant.'

    explainer = SHAP(model, tokenizer, device)
    evaluator = Evaluator(explainer)
    infidelity = evaluator.get_local_infidelity_mask_top_k(input2, 1)
    print(infidelity)
