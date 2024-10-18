import lime.lime_text
import shap
import lime

class Explainer():
    """Wrapper around model allowing for quick explanations"""
    def __init__(self, model):
        assert model.__class__.__bases__ in ["PreTrainedModel"], "Currently only HF transformers are supported"
        self.model = model

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
            raise ValueError(f'Unrecognized method {method}. 
                             Choose one of {recognized_methods}')
        
        if method == "shap":
            explainer = shap.Explainer(self.model)
            shap_values = explainer(input)
            return shap_values
        
        if method == "lime":
            labels = self.model.config.label2id.keys()
            explainer = lime.lime_text.LimeTextExplainer(class_names=labels)
            exp = explainer.explain_instance(input, self.model)
            return exp.as_list

