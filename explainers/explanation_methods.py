""" 
    Explanation methods implementing `Explainer` abstract base class
"""
import numpy as np
import torch

import lime.lime_text
import shap
import lime
from torch.nn.functional import softmax
from transformers import pipeline
from copy import deepcopy

from explainers.explainer import Explainer
from utils.utils import get_device


class SHAP(Explainer):
    def __init__(self, model, tokenizer, device=None):
        if not device:
            device = get_device()
        super().__init__(model, tokenizer, device)
        self.explainer = pipeline('text-classification',
                                  model=self.model,
                                  device=device,
                                  tokenizer=self.tokenizer,
                                  top_k=None)

    def explain(self, input):
        explainer = shap.Explainer(self.explainer)
        shap_values = explainer([input]).values

        return shap_values[0]


class LIME(Explainer):
    def __init__(self, model, tokenizer, device=None):
        if not device:
            device = get_device()
        super().__init__(model, tokenizer, device)
        labels = self.model.config.label2id.keys()
        self.explainer = lime.lime_text.LimeTextExplainer(class_names=labels)

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
        exp = self.explainer.explain_instance(input, self.predict_proba)

        return exp.as_list()


class IG(Explainer):
    def __init__(self, model, tokenizer, device=None):
        if not device:
            device = get_device()
        super().__init__(model, tokenizer, device)

    def explain(self, input):
        input = self.tokenizer(input, return_tensors='pt',
                               padding=True, truncation=True)
        pred_label_idx = torch.argmax(
            self.model(**input).logits, dim=-1).item()
        ig, _ = self.integrated_gradients(
            input, pred_label_idx, self.predictions_and_gradients)

        return ig.tolist()[0]

    def predictions_and_gradients(self, input, target_label_idx):
        # Don't update model params
        self.model.eval()

        # Must compute gradients w.r.t embeddings for discrete token input
        embeddings = self.model.get_input_embeddings()(
            input['input_ids'])  # : [Bs, input_ids, emb]
        embeddings = embeddings.detach().requires_grad_()
        logits = self.model(inputs_embeds=embeddings).logits

        target_class = logits[0][target_label_idx]
        target_class.backward()

        # Average over embedding dimension to compute gradient for input token
        gradients = np.average(embeddings.grad, axis=-1)
        predictions = torch.softmax(logits, dim=-1)

        return predictions, gradients

    def integrated_gradients(
            self,
            inp,
            target_label_index,
            predictions_and_gradients,
            baseline=None,
            steps=50):
        """
        adapted from: 
        https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py

        Computes integrated gradients for a given network and prediction label.

        This method only applies to classification networks, i.e., networks 
        that predict a probability distribution across two or more class labels.

        Access to the specific network is provided to the method via a
        'predictions_and_gradients' function provided as argument to this method.
        The function takes a batch of inputs and a label, and returns the
        predicted probabilities of the label for the provided inputs, along with
        gradients of the prediction with respect to the input. Such a function
        should be easy to create in most deep learning frameworks.

        Args:
            inp: The specific input for which integrated gradients must be computed.
            target_label_index: Index of the target class for which integrated gradients
            must be computed.
            predictions_and_gradients: This is a function that provides access to the
            network's predictions and gradients. It takes the following
            arguments:
            - inputs: A batch of tensors of the same same shape as 'inp'. The first
                dimension is the batch dimension, and rest of the dimensions coincide
                with that of 'inp'.
            - target_label_index: The index of the target class for which gradients
                must be obtained.
            and returns:
            - predictions: Predicted probability distribution across all classes
                for each input. It has shape <batch, num_classes> where 'batch' is the
                number of inputs and num_classes is the number of classes for the model.
            - gradients: Gradients of the prediction for the target class (denoted by
                target_label_index) with respect to the inputs. It has the same shape
                as 'inputs'.
            baseline: [optional] The baseline input used in the integrated
            gradients computation. If None (default), the all zero tensor with
            the same shape as the input (i.e., 0*input) is used as the baseline.
            The provided baseline and input must have the same shape. 
            steps: [optional] Number of intepolation steps between the baseline
            and the input used in the integrated gradients computation. These
            steps along determine the integral approximation error. By default,
            steps is set to 50.

        Returns:
            integrated_gradients: The integrated_gradients of the prediction for the
            provided prediction label to the input. It has the same shape as that of
            the input.

            The following output is meant to provide debug information for sanity
            checking the integrated gradients computation.
            See also: sanity_check_integrated_gradients

            prediction_trend: The predicted probability distribution across all classes
            for the various (scaled) inputs considered in computing integrated gradients.
            It has shape <steps, num_classes> where 'steps' is the number of integrated
            gradient steps and 'num_classes' is the number of target classes for the
            model.
        """
        if baseline is None:
            baseline = deepcopy(inp)
            baseline['input_ids'] = torch.zeros_like(baseline['input_ids'])

        # Scale input and compute gradients.
        scaled_inputs = []
        scaled_input = deepcopy(inp)
        for i in range(0, steps+1):
            tmp = (baseline['input_ids'] + (float(i)/steps) *
                   (inp['input_ids'] - baseline['input_ids'])).int()
            scaled_inputs.append(tmp)
        scaled_input['input_ids'] = torch.stack(
            scaled_inputs, dim=0).squeeze(1)
        predictions, grads = predictions_and_gradients(
            scaled_input, target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>

        # Use trapezoidal rule to approximate the integral.
        # See Section 4 of the following paper for an accuracy comparison between
        # left, right, and trapezoidal IG approximations:
        # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
        # https://arxiv.org/abs/1908.06214
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = np.average(grads, axis=0)
        integrated_gradients = (
            # shape: <inp.shape>
            inp['input_ids']-baseline['input_ids'])*avg_grads

        return integrated_gradients, predictions


if __name__ == "__main__":
    from transformers import AutoTokenizer, BertForSequenceClassification
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")
    model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity")
    model = model.to(device)
    id2label = model.config.id2label
    ig_explainer = IG(model, tokenizer, device)
    input = 'I seriously LOVE this great movie. This part has nothing to do with how I feel.'
    output = model(
        **(tokenizer(input, return_tensors='pt', padding=True, truncation=True)))
    print(output)
    out_class = id2label[torch.argmax(
        torch.softmax(output.logits, dim=-1), dim=-1).item()]
    print(f"Prediction:{out_class}\nExplanation:{ig_explainer.explain(input)}")
