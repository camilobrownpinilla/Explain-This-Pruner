"""
    Abstract evaluator base class
    Defines abstract methods that must be implemented
    by children classes (e.g. INFID, FCOR)
"""

from abc import ABC, abstractmethod
import torch
import heapq

from explainers.explainer import Explainer


class FaithfulnessEvaluator(ABC):
    """Wrapper around explainer for evaluating explanation methods"""

    def __init__(self, explainer : Explainer):
        self.model = explainer.model   # the 'black box' being explained
        self.tokenizer = explainer.tokenizer
        self.explainer = explainer  # explanation method to evaluate
        self.device = explainer.device
        # explanation method used to produce `explanation`
        self.method = type(self.explainer).__name__
        self.MASK = self.tokenizer.mask_token_id  # id of [MASK] token
    
    @abstractmethod
    def get_local_faithfulness(self, input : str, method : str, k : int, iters : int):
        """
        Returns local faithfulness of explanation method at `input` with respect to the model
        Input is perturbed using `method`

        params: 
            input: input being explained
            method: the method used to perturb the input
            k: number of features to mask
            iters: if method is k_subset, the number of random k-subsets to evaluate

        returns:
            local faithfulness
        """
        pass
    
    def evaluate_faithfulness(self, dataset, method : str, k : int, ptg : float = 0.2):
        """
        Returns the average local faithfulness of the explanation method over `ptg` percent of test samples in `dataset`.
        Computes faithfulness by masking features of an input sample using `method`.

        params:
            dataset: a dataset from data.datasets
            method: method used to perturb input
            k: number of features to mask
            ptg: percentage of test set to evaluate on. default 20%

        returns:
            average faithfulness score
        """
        # shuffle test set for random sampling
        test_set = dataset.test()
        shuffled_set = test_set.shuffle(seed=4)  # set seed for reproducibility
        infid = 0
        num_samples = int(len(shuffled_set) * ptg)
        print(f'evaluating infidelity on {num_samples} test samples')
        for sample in shuffled_set[dataset.x][:num_samples]:
            infid += self.get_local_faithfulness(sample, k, method)

        return infid / num_samples
    
    def eval_perturbation(self, tokenized_input, predicted_class_id, explanation, feature_ids):
        """
        Masks features of `tokenized_input` at `feature_ids`.
        Computes sum of masked feature importances and output of model on perturbed input.

        params:
            tokenized_input: tokenized input being explained
            predicted_class_id: index of predicted class for input being explained
            explanation: a feature-attribution explanation of the tokenized input
            feature_ids: list of indices of features to be masked

        returns:
            sum of masked feature importances, output of model on perturbed input (logits)
        """
        # store copy of original tokens
        tokens = tokenized_input["input_ids"].clone()

        # perturb input by masking the tokens at `feature_ids`
        perturbed_tokens = tokenized_input["input_ids"]
        perturbed_tokens[0, feature_ids] = self.MASK
        tokenized_input["input_ids"] = perturbed_tokens

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
            
        return importance_sum, perturbed_logits
    
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
        if self.method == 'SHAP':
            # `explanation` is a 2d array with elts:
            # [contribution to label 0, contribution to label 1]
            top_feature_ids = heapq.nlargest(
                k, range(len(explanation)), key=lambda i: explanation[i][predicted_class_id])

        elif self.method == 'LIME':
            # `explanation` is a list [(token, importance),...]
            # negative feature importance -> feature contributes to label 0
            # positive feature importance -> feature contributes to label 1
            if predicted_class_id == 0:
                # Get the indices of the k smallest importance values (contributing to label 0)
                top_feature_ids = heapq.nsmallest(
                    k, range(len(explanation)), key=lambda i: explanation[i][1])
            else:
                # Get the indices of the k largest importance values (contributing to label 1)
                top_feature_ids = heapq.nlargest(
                    k, range(len(explanation)), key=lambda i: explanation[i][1])

        elif self.method == 'IG':
            # `explanation` is a list of feature importance scores w.r.t. model prediction
            top_feature_ids = heapq.nlargest(
                k, range(len(explanation)), key=lambda i: explanation[i])

        else:
            raise ValueError(f'Explanation method {self.method} not supported')

        return top_feature_ids
