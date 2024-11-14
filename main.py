import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from explainers.explanation_methods import SHAP, LIME, IG
from evaluations.evaluator import Evaluator
from data.datasets import IMDB, Emotion, YelpPolarity


def eval_models(path, tokenizer, explainers, test_set, device, ptg=0.05, k=1):
    """
    For each model in `path`, evaluates infidelity of each explainer on `test_set`.
    For each explainer, creates and saves a bar chart of infidelity scores for each model.

    params:
        path: path to directory of models, e.g. './ckpts'
        tokenizer: tokenizer used in training models at `path`
        explainers: list of explainers, e.g. [SHAP, LIME, IG]
        test_set: test set to evaluate explainability on
        device: device to evaluate models on
        ptg: percentage of test_set to evaluate on, default 5%
        k: Number of features to mask when computing infidelity 
    """
    # Load in models contained in `path`
    print(f"Loading models from {path}")
    models, arch = load_latest_checkpoints(path)

    print(f"Creating evaluators for explanations of {arch} models...")
    evaluators_dict = {}
    for explainer in explainers:
        evaluators_dict[explainer.__name__] = {}
        for name, model in models.items():
            evaluator = Evaluator(explainer(model, tokenizer, device))
            evaluators_dict[explainer.__name__][name] = evaluator

    for exp, evaluators in evaluators_dict.items():
        print(f'Evaluating infidelity of {exp} explanations...')
        infidelities = {}
        for model, eval in evaluators.items():
            infid = eval.evaluate_infidelity_mask_top_k(
                test_set, k=k, ptg=ptg)
            print(f"Infidelity of {exp} explanations of {model} model: {infid}")
            infidelities[model] = infid

        # Create bar chart
        plt.clf()
        plt.bar(range(len(infidelities)), infidelities.values(), color='pink', edgecolor='hotpink', alpha=0.7)
        plt.xticks(range(len(infidelities)), infidelities.keys())
        plt.xlabel('Model')
        plt.ylabel('Infidelity')
        plt.title(f'Infidelity of {exp} Explanations for Differently Pruned Versions of {arch}')
        
        # Save bar chart
        model_id = os.path.relpath(path, start='./')
        save_path = f'results/figs/{model_id}/'
        ext = f'{exp}_top_{k}.png'

        # Ensure unique filename
        file_path = save_path + ext
        counter = 1
        # while os.path.exists(file_path):
        #     file_path = f'{save_path}_{counter}{ext}'
        #     counter += 1
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(file_path, format='png', dpi=300)
        print(f"Chart saved to {file_path}")


def load_latest_checkpoints(path):
    """
    Returns models loaded from their latest checkpoint in the subdirs of `path`.
    
    E.g. if path = './ckpts', we load models from:
        ./ckpts/base, ./ckpts/smaller, ./ckpts/pruning_method_name

    params:
        path: path to directory of models, e.g. './ckpts'

    returns:
        models : dict, e.g. models['l1unstruct'] = model
        arch : architecture of models in `path`, e.g. BertForSequenceClassification
    """
    models = {}
    arch = None
    
    # Check if the path exists and is a directory
    if not os.path.isdir(path):
        raise FileNotFoundError(f"The directory '{path}' does not exist or is not a directory.")
        
    # Iterate over each subdirectory in path
    for model_dir in os.listdir(path):
        model_path = os.path.join(path, model_dir)

        # Check if it's a directory
        if os.path.isdir(model_path):
            # Load the model from the directory
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                models[model_dir] = model
                print(f"Loaded model from {model_path}")
                # Get name of model architecture
                if not arch:
                    arch = model.config.model_type
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")

    return models, arch


if __name__ == '__main__':
    top_path = './roBERTa_IMDB'
    paths = [os.path.join(top_path, sub_path) for sub_path in os.listdir(top_path)]
    tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    explainers = [SHAP, IG]
    dataset = IMDB()
    device = torch.device('cuda')
    
    for path in paths:
        for k in [1, 5, 10]:
            eval_models(path, tokenizer, explainers, dataset, device, k=k, ptg=0.01)
