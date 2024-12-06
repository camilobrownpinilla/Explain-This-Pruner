import os
import torch
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import json
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import cmasher as cmr

from explainers.explanation_methods import SHAP, LIME, IG
from evaluators.evaluation_methods import INFID, FCOR
from data.datasets import IMDB, Emotion, YelpPolarity


def eval_models(path, tokenizer, explainers, test_set, metric, device, ptg=0.05, k=1):
    """
    For each model in `path`, evaluates faithfulness of each explainer on `test_set`.
    For each explainer, creates and saves a bar chart of faithfulness scores for each model.

    params:
        path: path to directory of models, e.g. './ckpts'
        tokenizer: tokenizer used in training models at `path`
        explainers: list of explainers, e.g. [SHAP, LIME, IG]
        test_set: test set to evaluate explainability on
        device: device to evaluate models on
        ptg: percentage of test_set to evaluate on, default 5%
        k: Number of features to mask when computing faithfulness 
    """
    # Load in models contained in `path`
    print(f"Loading models from {path}")
    models, arch, accuracies = load_latest_checkpoints(path)
    
    if metric == 'infid':
        eval_metric = INFID
        metric_name = 'Infidelity'
    elif metric == 'fcor':
        eval_metric = FCOR
        metric_name = 'FCor'
    else:
        raise ValueError("metric must be 'infid' or 'fcor'")

    print(f"Creating evaluators for explanations of {arch} models...")
    evaluators_dict = {}
    for explainer in explainers:
        evaluators_dict[explainer.__name__] = {}
        for name, model in models.items():
            evaluator = eval_metric(explainer(model, tokenizer, device))
            evaluators_dict[explainer.__name__][name] = evaluator

    for exp, evaluators in evaluators_dict.items():
        print(f'Evaluating {metric_name} of {exp} explanations...')
        faithfulness = {}
        distributions = {}
        for model, eval in evaluators.items():
            f, d = eval.evaluate_faithfulness(
                test_set, method='k_subset', k=k, ptg=ptg)
            print(f"{metric_name} of {exp} explanations of {model} model: {f}")
            faithfulness[model] = f
            distributions[model] = d

       # Normalize accuracies
        norm = Normalize(vmin=min(accuracies.values()), vmax=max(accuracies.values()))
        cmap = cm['cmr.bubblegum']
        colors = [cmap(norm(acc)) for acc in accuracies.values()]

        # Create figure and axes with space for colorbar
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['base', 'RandUnstructured', 'L1Unstructured', 'L1Structured', 'smaller']
        fcors = [faithfulness[model] for model in models]
            
        # Create bar chart
        bars = ax.bar(range(len(models)), fcors, 
                    color=colors, edgecolor='grey', alpha=0.7)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} on {exp} Explanations for Differently Pruned Versions of {arch}')

        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Accuracy')

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save bar chart
        model_id = os.path.relpath(path, start='./')
        save_path = f'results/{ptg=}/{metric}/{model_id}/top_{k}/'
        ext = f'{exp}.png'
        file_path = save_path + ext
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(file_path, format='png', dpi=300)
        print(f"Chart saved to {file_path}")
        plt.close()

        # Plot faithfulness distributions
        plt.clf()
        plt.style.use('default')  # Use matplotlib default style
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create violin plot
        positions = range(1, len(distributions) + 1)
        violin_parts = ax.violinplot(list(distributions.values()), positions, 
                                   showmeans=True, showmedians=False)
        violin_parts['cbars'].set_color('white'), violin_parts['cmins'].set_color('white')
        violin_parts['cmaxes'].set_color('white')        
        # Color violins based on accuracy
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('grey')
            pc.set_alpha(1)
        
        # Color mean points
        violin_parts['cmeans'].set_color('white')
        
        # Customize plot
        ax.set_xticks(positions)
        ax.set_xticklabels(list(distributions.keys()), rotation=45, ha='right')
        ax.set_xlabel('Model')
        ax.set_ylabel(f'{metric_name} scores')
        ax.set_title(f'Distributions of {metric_name} on {exp} Explanations\nfor Differently Pruned Versions of {arch}')
        
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Accuracy')
        
        # Adjust layout
        plt.tight_layout()
        file_path = save_path + f'{exp}_distribution.png'
        plt.savefig(file_path, format='png', dpi=300)
        plt.close()
        print(f'Distribution plot saved to {file_path}')

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
    accuracies = {}
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
                model.to(device)
                models[model_dir] = model
                print(f"Loaded model from {model_path}")
                # Get name of model architecture
                if not arch:
                    arch = 'roBERTa' if model.config.model_type == 'bert' else 'distilBERT'

                # Get model accuracy
                with open(os.path.join(model_path, 'all_results.json'), 'r') as f:
                    data = json.load(f)
                    accuracies[model] = data['eval_accuracy']
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")

    return models, arch, accuracies


if __name__ == '__main__':
    top_path = './generated_models/roBERTa_YELP_3EP'
    paths = [os.path.join(top_path, sub_path) for sub_path in os.listdir(top_path)]
    tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    # tokenizer = AutoTokenizer.from_pretrained(
    # "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    explainers = [SHAP, IG]
    dataset = YelpPolarity()
    device = torch.device('cuda')
    metric = 'fcor'
    
    for path in paths:
        for k in [3]:
            eval_models(path, tokenizer, explainers, dataset, metric, device, k=k, ptg=0.03)
