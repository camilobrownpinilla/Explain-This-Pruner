import torch
from itertools import combinations
from transformers import AutoTokenizer, BertForSequenceClassification

from explainers.explainer import Explainer
from explainers.explanation_methods import SHAP, LIME, IG


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
        # explanation method used to produce `explanation`
        self.method = type(self.explainer).__name__
        self.MASK = self.tokenizer.mask_token_id  # id of [MASK] token

    def evaluate_infidelity_mask_top_k(self, dataset, k=1, ptg=0.2):
        """
        Returns the average local infidelity of the explanation method over `num_samples` rand test samples in `dataset`.
        Computes local infidelity by masking top k features of an input sample.

        params:
            dataset: a dataset from data.datasets
            k: number of top features to mask
            ptg: percentage of test set to evaluate on. default 20%

        returns:
            average infidelity
        """

        # shuffle test set for random sampling
        test_set = dataset.test()
        shuffled_set = test_set.shuffle(seed=4)  # set seed for reproducibility
        infid = 0
        num_samples = int(len(shuffled_set) * ptg)
        print(f'evaluating infidelity on {num_samples} test samples')
        for sample in shuffled_set[dataset.x][:num_samples]:
            infid += self.get_local_infidelity_mask_top_k(sample, k)

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

        top_feature_ids = self.get_top_k_features(
            explanation, predicted_class_id, k)

        # perturb input by masking the tokens at `top_feature_ids` and evaluate for infidelity
        infid = self.eval_perturbation(
            tokenized_input, predicted_class_id, logits, explanation, top_feature_ids)

        return infid

    def get_top_k_features(self, explanation, predicted_class_id, k=1):
        """
        Gets top k features with greatest contribution to predicted class

        params:
            explanation: a feature-attribution explanation of some sample
            predicted_class_id: index of predicted class for sample being explained
            k: number of top features to get

        returns:
            top_feature_ids: list of indices of top k features
        """
        if k >= len(explanation) or k < 1:
            raise ValueError(
                'k must be at least 1 and no greater than the number of features')

        if self.method == 'SHAP':
            # `explanation` is a 2d array with elts:
            # [contribution to label 0, contribution to label 1]
            top_feature_ids = sorted(
                range(len(explanation)), key=lambda i: explanation[i][predicted_class_id], reverse=True)[:k]

        elif self.method == 'LIME':
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

        elif self.method == 'IG':
            # `explanation` is a list of feature importance scores w.r.t. model prediction
            top_feature_ids = sorted(
                range(len(explanation)), key=lambda i: explanation[i], reverse=True)[:k]

        else:
            raise ValueError(f'Explanation method {self.method} not supported')

        return top_feature_ids

    def get_local_infidelity_mask_pairs(self, input):
        """
        Returns local infidelity of explanation method at input with respect to the model
        Input is perturbed by masking pairs of features, one pair at a time

        params: 
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

        # Get all distinct pairs of feature ids
        pairs = list(combinations(range(len(explanation)), 2))

        # Accumulate infidelities of all pair-maskings
        infid = 0
        for pair in pairs:
            infid += self.eval_perturbation(tokenized_input,
                                            predicted_class_id, logits, explanation, pair)

        return infid

    def eval_perturbation(self, tokenized_input, predicted_class_id, logits, explanation, feature_ids):
        """
        Masks features of `tokenized_input` at `feature_ids`.
        Measures change in output of model on perturbed input w.r.t. the importances of the masked features.
        Returns this local infidelity.

        params:
            tokenized_input: tokenized input being explained
            predicted_class_id: index of predicted class for input being explained
            logits: output of model on input
            explanation: a feature-attribution explanation of the tokenized input
            feature_ids: list of indices of features to be masked

        returns:
            local infidelity
        """
        # store copy of original tokens
        tokens = tokenized_input["input_ids"].clone()

        print(f'original tokens: {tokenized_input["input_ids"]}')
        print(f'logits: {logits}')
        print(f'predicted class id: {predicted_class_id}')
        print(f'explanation: {explanation}')
        print(f'masking following feature ids: {feature_ids}')

        # perturb input by masking the tokens at `feature_ids`
        perturbed_tokens = tokenized_input["input_ids"]
        perturbed_tokens[0, feature_ids] = self.MASK
        tokenized_input["input_ids"] = perturbed_tokens

        print(f'perturbed tokens: {tokenized_input["input_ids"]}')

        # get output of model on perturbed input
        with torch.no_grad():
            perturbed_logits = self.model(**tokenized_input).logits

        # reset original tokens
        tokenized_input["input_ids"] = tokens

        # get sum of importances of masked features
        if self.method == 'SHAP':
            importance_sum = sum(explanation[feature_ids, predicted_class_id])
        elif self.method == 'LIME':
            importance_sum = sum([explanation[i][1] for i in feature_ids])
        elif self.method == 'IG':
            importance_sum = sum([explanation[i] for i in feature_ids])

        print(f'importance sum: {importance_sum}')
        # calculate local infidelity
        # compares sum of masked feature importances to change in model output after perturbation
        infidelity = (importance_sum - (logits[0, predicted_class_id] -
                                        perturbed_logits[0, predicted_class_id]))**2

        return infidelity.item()


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")
    model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")
    device = torch.device('cpu')
    model = model.to(device)

    input1 = 'Hello, my dog is so terribly ugly'
    input2 = 'I am very happy about this restaurant.'

    explainer = IG(model, tokenizer, device)
    evaluator = Evaluator(explainer)
    infidelity = evaluator.get_local_infidelity_mask_top_k(input2, 3)
    print(infidelity)
