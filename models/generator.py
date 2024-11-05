from datasets import load_dataset
from transformers import Trainer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import evaluate
import numpy as np
from copy import deepcopy
import os

from pruners.pruning_methods import RandUnstructured, L1Unstructured, CustomMask
from utils.utils import reinitialize_weights, get_params_to_prune


def generate(model, tokenizer, data, encode_fn, pruning_methods, prune_ptg,
             save_dir='../ckpts/', train_epochs=1):
    """
    Given a model, tokenizer, data, and pruning method(s), generate, train, and 
    save stock model, smaller model, and pruned model(s). Model performance is 
    also saved.

    Args:
        model (PreTrainedModel): Base Huggingface model. 
        tokenizer (PreTrainedTokenizer): Huggingface tokenizer.
        data (str): Name of Huggingface dataset or path to local dataset loading 
                    script
        encode_fn: Function to encode dataset using tokenizer
        pruning_methods (list[Pruner]): List of pruning methods to use.
        prune_ptg (float): Percentage to prune model
        save_dir (str, optional): Where to save models. Defaults to '../ckpts/'.
    """
    HOT_PINK = "\033[95;1;4m"
    RESET = "\033[0m"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid forking

    # Load data
    raw_data = load_dataset(data)
    tokenized_data = raw_data.map(encode_fn)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Reset model to random inits
    base_model = model
    reinitialize_weights(base_model)

    # Train base model
    print(f'{HOT_PINK}Training base model...{RESET}')
    train_args = TrainingArguments(f'{save_dir}/base',
                                   eval_strategy='epoch',
                                   save_strategy='no',
                                   per_device_train_batch_size=64,
                                   per_device_eval_batch_size=64,
                                   num_train_epochs=train_epochs,
                                   fp16=True)
    base_trainer = Trainer(base_model,
                           train_args,
                           train_dataset=tokenized_data['train'],
                           eval_dataset=tokenized_data['test'],
                           data_collator=data_collator,
                           processing_class=tokenizer,
                           compute_metrics=_compute_accuracy)
    base_trainer.train()
    base_trainer.save_model(f'{save_dir}/base')

    # Create and train smaller model
    print(f'{HOT_PINK}Training smaller model...{RESET}')
    smaller_model = deepcopy(base_model)
    reinitialize_weights(smaller_model)

    smaller_model = _make_pruned_model(
        RandUnstructured, smaller_model, prune_ptg)
    train_args = TrainingArguments(f'{save_dir}/smaller',
                                   eval_strategy='epoch',
                                   save_strategy='no',
                                   per_device_train_batch_size=64,
                                   per_device_eval_batch_size=64,
                                   num_train_epochs=train_epochs,
                                   fp16=True)
    smaller_trainer = Trainer(smaller_model,
                              train_args,
                              train_dataset=tokenized_data['train'],
                              eval_dataset=tokenized_data['test'],
                              data_collator=data_collator,
                              processing_class=tokenizer,
                              compute_metrics=_compute_accuracy)
    smaller_trainer.train()
    smaller_trainer.save_model(f'{save_dir}/smaller')

    # Create and train pruned models
    print(f'{HOT_PINK}Training pruned models...{RESET}')
    base_model = AutoModelForSequenceClassification.from_pretrained(
        f'{save_dir}/base')  # Prune from trained base model
    for method in pruning_methods:
        method_name = method.__name__
        pruned_model = _make_pruned_model(method, base_model, prune_ptg)
        train_args = TrainingArguments(f'{save_dir}/{method_name}',
                                       eval_strategy='epoch',
                                       save_strategy='no',
                                       per_device_train_batch_size=64,
                                       per_device_eval_batch_size=64,
                                       num_train_epochs=train_epochs,
                                       fp16=True)
        smaller_trainer = Trainer(pruned_model,
                                  train_args,
                                  train_dataset=tokenized_data['train'],
                                  eval_dataset=tokenized_data['test'],
                                  data_collator=data_collator,
                                  processing_class=tokenizer,
                                  compute_metrics=_compute_accuracy)
        smaller_trainer.train()
        smaller_trainer.save_model(f'{save_dir}/{method_name}')

    # Give user save locations
    print(f'{HOT_PINK}Models generated and saved to {save_dir}{RESET}')


def _compute_accuracy(preds):
    metric = evaluate.load('accuracy')
    logits, labels = preds
    predictions = np.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


def _make_pruned_model(method, base_model, prune_ptg):
    pruner = method()
    pruned_model = deepcopy(base_model)
    pruner.prune(get_params_to_prune(pruned_model), prune_ptg)
    return pruned_model
