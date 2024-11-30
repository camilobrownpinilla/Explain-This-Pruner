""" 
    Faithfulness evaluation methods implementing `FaithfulnessEvaluator` abstract base class
"""

import torch
import numpy as np
import random
from transformers import AutoTokenizer, BertForSequenceClassification

from evaluators.evaluator import FaithfulnessEvaluator
from explainers.explanation_methods import SHAP, LIME, IG


class FCOR(FaithfulnessEvaluator):
    """Implementation of ..."""
    
    def __init__(self, explainer):
        super().__init__(explainer)
        
    def evaluate_fcor(self, dataset, method, k, ptg=0.2):
        return self.evaluate_faithfulness(dataset, method, k, ptg)
    
    def get_local_faithfulness(self, input, method, k, iters=10):
        return self.get_local_fcor(input, method, k, iters)
    
    def get_local_fcor(self, input, method, k, iters=10):
        max_length = self.model.config.max_position_embeddings
        tokenized_input = self.tokenizer(input, 
                                         return_tensors="pt", 
                                         truncation=True,
                                         max_length=max_length).to(self.device)

        with torch.no_grad():
            logits = self.model(**tokenized_input).logits

        predicted_class_id = logits.argmax().item()
        explanation = self.explainer.explain(input)
        
        # Print the decoded tokens in order of importance to predicted class
        
        # importances = explanation #[:, predicted_class_id]
        # token_ids = tokenized_input['input_ids']
        # # Flatten the tensor to get a list of token IDs
        # token_ids_list = token_ids.squeeze().tolist()

        # # Pair token IDs with their importances
        # token_importance_pairs = list(zip(token_ids_list, importances))

        # # Sort by importance in descending order
        # sorted_pairs = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)

        # # Decode and print tokens in order of importance
        # for token_id, importance in sorted_pairs:
        #     token = tokenizer.decode([token_id])
        #     print(f"{token}: {importance}")
        
        if k < 1 or k > len(explanation):
            raise ValueError(
                'k must be at least 1 and no greater than the number of features')
        
        importance_sums = []
        logits_delta = []
        if method == 'k_subset':
            # perturb input by masking a random subset of k features
            ids = range(len(explanation))
            for _ in range(iters):
                # randomly sample k feature ids for perturbation
                subset = random.sample(ids, k)
                
                importance_sum, perturbed_logits = self.eval_perturbation(
                    tokenized_input, predicted_class_id, explanation, subset
                )
                delta = logits[0, predicted_class_id] - perturbed_logits[0, predicted_class_id]
                importance_sums.append(importance_sum)
                logits_delta.append(delta)
            
            # get correlation between sum of masked feature importances and change in model output
            fcor = np.corrcoef(importance_sums, logits_delta)[0, 1]
            
        else:
            raise ValueError('supported perturbation methods: k_subset')
        
        return fcor


class INFID(FaithfulnessEvaluator):
    """Implementation of Yeh et al.'s infidelity metric for faithfulness evaluation"""
    
    def __init__(self, explainer):
        super().__init__(explainer)
    
    def evaluate_infidelity(self, dataset, method, k, ptg=0.2):
        return self.evaluate_faithfulness(dataset, method, k, ptg)
    
    def get_local_faithfulness(self, input, method, k, iters=5):
        return self.get_local_infidelity(input, method, k, iters)
        
    def get_local_infidelity(self, input, method, k, iters=5):
        max_length = self.model.config.max_position_embeddings
        tokenized_input = self.tokenizer(input, 
                                         return_tensors="pt", 
                                         truncation=True,
                                         max_length=max_length).to(self.device)

        with torch.no_grad():
            logits = self.model(**tokenized_input).logits

        predicted_class_id = logits.argmax().item()
        explanation = self.explainer.explain(input)
        
        if k < 1 or k > len(explanation):
            raise ValueError(
                'k must be at least 1 and no greater than the number of features')
        
        infid = 0
        if method == 'top_k':
            # perturb input by masking top k features
            top_feature_ids = self.get_top_k_features(
                explanation, predicted_class_id, k)
            infid = self.eval_perturbation_infid(
                tokenized_input, predicted_class_id, logits, explanation, top_feature_ids)
            
        elif method == 'k_subset':
            # perturb input by masking a random subset of k features
            ids = range(len(explanation))
            for _ in range(iters):
                # randomly sample k feature ids for perturbation
                subset = random.sample(ids, k)
                # mask the subset and evaluate infidelity
                infid += self.eval_perturbation_infid(tokenized_input,
                                                predicted_class_id, logits, explanation, subset)
            infid /= iters
            
        else:
            raise ValueError('supported perturbation methods: top_k, k_subset')
        
        return infid
            
    def eval_perturbation_infid(self, tokenized_input, predicted_class_id, logits, explanation, feature_ids):
        """
        Masks features of `tokenized_input` at `feature_ids`.
        Measures change in output of model on perturbed input w.r.t. the importances of the masked features.
        Returns this local infidelity (adopted from Yeh et al.).

        params:
            tokenized_input: tokenized input being explained
            predicted_class_id: index of predicted class for input being explained
            logits: output of model on input
            explanation: a feature-attribution explanation of the tokenized input
            feature_ids: list of indices of features to be masked

        returns:
            local infidelity
        """
        importance_sum, perturbed_logits = self.eval_perturbation(
            tokenized_input, predicted_class_id, explanation, feature_ids
        )
        # calculate local infidelity (Yeh et al.)
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

    explainer = IG(model, tokenizer, device)
    evaluator = FCOR(explainer)
    
    inputs = ['Hello, my dog is so terribly ugly',
              'I am very happy about this restaurant.',
              'This food is absolutely delicious.',
              'This food is so disgusting and nasty, but healthy.']

    for input in inputs[-1:]:
        print(evaluator.get_local_fcor(input, 'k_subset', 1, iters=3))
