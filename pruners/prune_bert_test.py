from transformers import BertForSequenceClassification
from pruning_methods import Pruner, PruningMethod
from pruners.test_suite import *
from pruners.finetuner import *

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity")
print('model loaded')

# List to store all parameters to prune
paras_to_prune = []
# Loop through all encoder layers to collect dense weights and add them to paras_to_prune list
# Note: We are only pruning the weights, not biases
# Note: We are not pruning the embeddings, pooler, classifier, layer norm, activation, or dropout layers
for layer in model.bert.encoder.layer:
    # Attention weights (query, key, value, and output projection)
    paras_to_prune.append((layer.attention.self.query, 'weight'))
    paras_to_prune.append((layer.attention.self.key, 'weight'))
    paras_to_prune.append((layer.attention.self.value, 'weight'))
    paras_to_prune.append((layer.attention.output.dense, 'weight'))

    # Intermediate dense layer
    paras_to_prune.append((layer.intermediate.dense, 'weight'))

    # Output dense layer
    paras_to_prune.append((layer.output.dense, 'weight'))


# Initialize Pruner instance
Bert_pruner = Pruner(model)
# Initialize PruningMethod instances
randomly_prune_Bert = PruningMethod(
    type="RandomUnstructured", paras_to_prune=paras_to_prune, percentage=0.1, mask=None)
L1_prune_Bert = PruningMethod(
    type="L1Unstructured", paras_to_prune=paras_to_prune, percentage=0.9, mask=None)
custom_prune_Bert = PruningMethod(
    type="Custom", paras_to_prune=paras_to_prune, percentage=None, mask=[])

# Prune the model
Bert_pruner.prune(L1_prune_Bert)


test_sparsity(paras_to_prune)

Bert_fine_tuner(model)
