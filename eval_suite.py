"""
Evaluates local infidelity of SHAP, LIME, and/or IG explanations of 
BertForSequenceClassification (pretrained on Yelp polarity) on 6 sample inputs
Visualizes results in bar chart for differently pruned versions of the model
Pruned model accuracy has NOT been evaluated, and pruned models are not fine-tuned in this script
"""

import matplotlib.pyplot as plt
from pruners.pruning_methods import L1Unstructured, RandUnstructured
import torch
from evaluations.evaluator import Evaluator
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification
from explainers.explanation_methods import SHAP, LIME, IG
import torch
from copy import deepcopy


# Returns a list of model params given a bert `model`
def get_params_to_prune(model):
    params_to_prune = []
    for layer in model.bert.encoder.layer:
        # Attention weights (query, key, value, and output projection)
        params_to_prune.append((layer.attention.self.query, 'weight'))
        params_to_prune.append((layer.attention.self.key, 'weight'))
        params_to_prune.append((layer.attention.self.value, 'weight'))
        params_to_prune.append((layer.attention.output.dense, 'weight'))

        # Intermediate dense layer
        params_to_prune.append((layer.intermediate.dense, 'weight'))

        # Output dense layer
        params_to_prune.append((layer.output.dense, 'weight'))

    return params_to_prune


# Returns a dictionary of infidelity scores for different
# explanations of differently pruned models
# `explainers` is list of explainer classes
# output: { prune_method: {explanation_method: infidelity}... }
def eval_suite(model, tokenizer, inputs, prune_ptg, explainers):
    model.to(device)
    # Init models and grab model params to prune
    randunstructured_model = deepcopy(model).to(device)
    l1unstructured_model = deepcopy(model).to(device)

    randunstruct_params = get_params_to_prune(randunstructured_model)
    l1unstruct_params = get_params_to_prune(l1unstructured_model)

    # Initialize pruners and make pruned models
    print('Pruning models...')
    unpruned_model = model

    randunstructured_pruner = RandUnstructured()
    l1unstructured_pruner = L1Unstructured()
    randunstructured_pruner.prune(randunstruct_params, prune_ptg)
    l1unstructured_pruner.prune(l1unstruct_params, prune_ptg)

    # e.g. evaluators_dict['randunstruct'] = {'SHAP': Evaluator(), 'LIME': Evaluator(), 'IG': Evaluator()}
    evaluators_dict = {'randunstruct': {}, 'l1unstruct': {}, 'unpruned': {}}
    for explainer in explainers:
        randunstruct_explainer = explainer(
            randunstructured_model, tokenizer, device)
        l1unstruct_explainer = explainer(
            l1unstructured_model, tokenizer, device)
        unpruned_explainer = explainer(unpruned_model, tokenizer, device)

        evaluators_dict['randunstruct'][explainer.__name__] = Evaluator(
            randunstruct_explainer)
        evaluators_dict['l1unstruct'][explainer.__name__] = Evaluator(
            l1unstruct_explainer)
        evaluators_dict['unpruned'][explainer.__name__] = Evaluator(
            unpruned_explainer)

    infidelities = {}
    for input in tqdm(inputs, desc='Evaluating', unit='input'):
        for prune_method, evaluator_set in evaluators_dict.items():
            # Initialize prune_method in infidelities if not present
            if prune_method not in infidelities:
                infidelities[prune_method] = {}

            for expla_method, evaluator in evaluator_set.items():
                # Initialize expla_method as a list if not present
                if expla_method not in infidelities[prune_method]:
                    infidelities[prune_method][expla_method] = []

                # Append the result of get_local_infidelity to the list
                infidelities[prune_method][expla_method].append(
                    evaluator.get_local_infidelity(input, k=1))

    return infidelities


device = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity")
model = model.to(device)

inputs = ['Camilo CANNOT CODE FOR HIS LIFE. I DONT LIKE HIM!!!!',
          'David is GREAT at soccer. Can recommend <thumbs up>!',
          'Joey is joey. I feel very neutrally about him',
          'Finale is the best professor Harvard has EVER had. Would recommend!',
          'I AM GOING TO SCREAMMMMMMMMMMM AHHHHHHHHHHHH',
          'Paula and Hiwot are great TFs!']


explainers = [SHAP]
infidelities = eval_suite(model, tokenizer, inputs, .20, explainers)


print(infidelities)

method = 'SHAP'
categories = ['unpruned', 'l1unstruct', 'randunstruct']
vals1 = [infidelities[c][method][0] for c in categories]
vals2 = [infidelities[c][method][3] for c in categories]
vals3 = [infidelities[c][method][5] for c in categories]

# Create a 3x1 subplot layout
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
xlabels = ['Unpruned', 'L1 Unstructured', 'Random Unstructured']

# Set an overarching title for the entire figure
fig.suptitle(f'Local Infidelity of {method} Explanations for Models Pruned via Different Methods',
             fontsize=20, fontweight='bold')

# Plot each subplot with individual data and titles
axs[0].bar(xlabels, vals1, color='skyblue')
axs[1].bar(xlabels, vals2, color='salmon')
axs[2].bar(xlabels, vals3, color='lightgreen')

for i in range(len(categories)):
    axs[i].set_title(f'input: "{inputs[i]}"')
    axs[i].set_ylabel('Infidelity Score')


axs[2].set_xlabel('Pruning Method')
plt.tight_layout(rect=[0, 0, 1, 0.96])


plt.show()
