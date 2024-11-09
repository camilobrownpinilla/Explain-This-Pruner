# specify:
#   model, tokenizer, data, encode_fn, pruning_methods, prune_ptg

# generate models using above args

# load generated models from ./ckpts:
#    these are ./ckpts/base, ./ckpts/smaller, ./ckpts/pruning_method_name

# specify explainers to evaluate, e.g. [SHAP, IG]
# create a dict of Evaluator objects, e.g.:
#     evaluators_dict['l1unstruct'] = {'SHAP': Evaluator(), 'LIME': Evaluator(), 'IG': Evaluator()}
# this dict structure may make more sense actually:
#     evaluators_dict['SHAP'] = {'base': Evaluator(), 'smaller': Evaluator(), 'l1unstruct': Evaluator(),...}
#     i.e. evaluators_dict[explainer][model] = Evaluator

# using samples from the test split of the dataset on which above models were trained:
#     evaluate the infidelity of each explainer on each of above models

# visualize results
# one bar chart per explainer:
#     display infidelity score of that explainer on each model


from explainers.explainer import Explainer
from explainers.explanation_methods import SHAP
from evaluations.evaluator import Evaluator

# model
# tokenizer
# test_set
# device

explainer = SHAP(model, tokenizer, device)
evaluator = Evaluator(explainer)

infid = evaluator.evaluate_infidelity_mask_top_k(test_set, k=1, ptg=0.05)
