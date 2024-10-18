import lime.lime_text
import shap
import lime
from transformers.modeling_utils import PreTrainedModel
from torch.nn.functional import softmax

class Explainer():
    """Wrapper around model allowing for quick explanations"""
    def __init__(self, model, tokenizer):
        assert isinstance(model, PreTrainedModel),\
            "Currently only HF transformers are supported"
        self.model = model
        self.tokenizer = tokenizer

    def explain(self, method: str, input):
        """
        Method to explain model's prediction of 'input'

        params: 
            method (str): Explanation method to use
            input (any): input to be explained

        returns:
            explanation
        """
        recognized_methods = ["shap", "lime"]
        if method not in recognized_methods:
            raise ValueError(f'''Unrecognized method {method}. 
                             Choose one of {recognized_methods}''')
        
        if method == "shap":
            explainer = shap.Explainer(self.model)
            shap_values = explainer(input)
            return shap_values
        
        if method == "lime":
            labels = self.model.config.label2id.keys()
            explainer = lime.lime_text.LimeTextExplainer(class_names=labels)
            exp = explainer.explain_instance(input, self.predict_proba)
            return exp.as_list()
    
    def predict_proba(self, texts):
        """
        Prediction function for LIME explainer

        params:
            texts (list of str): List of input texts

        returns:
            predictions (numpy.ndarray): Array of prediction probabilities
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        probabilities = softmax(outputs.logits, dim=-1).detach().numpy()
        return probabilities

