from transformers import Trainer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import evaluate
import numpy as np
from copy import deepcopy
import os

from pruners.pruning_methods import RandUnstructured, L1Unstructured, CustomMask
from utils.utils import reinitialize_weights, get_params_to_prune


def generate(model, tokenizer, dataset, pruning_methods, prune_ptg, 
             save_dir='../ckpts/', train_epochs=1):
    """
    Given a model, tokenizer, data, and pruning method(s), generate, train, and 
    save stock model, smaller model, and pruned model(s). Model performance is 
    also saved.

    Args:
        model (PreTrainedModel): Base Huggingface model. 
        tokenizer (PreTrainedTokenizer): Huggingface tokenizer.
        data (StandardDataset): Initialized Dataset of type StandardDataset.
        pruning_methods (list[Pruner]): List of pruning methods to use.
        prune_ptg (float): Percentage to prune model
        save_dir (str, optional): Where to save models. Defaults to '../ckpts/'.
    """
    HOT_PINK = "\033[1;4;38;5;197m" 
    RESET = "\033[0m"
    BATCH_SIZE = 128
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid forking

    # Prepare data
    dataset.dataset = dataset.dataset.map(dataset.encode(tokenizer))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Reset model to random inits
    base_model = model
    reinitialize_weights(base_model)

    # Train base model
    print(f'{HOT_PINK}Training base model...{RESET}')
    train_args = TrainingArguments(f'{save_dir}/base', 
                                   eval_strategy='epoch', 
                                   logging_strategy='epoch',
                                   save_strategy='no',
                                   per_device_train_batch_size=BATCH_SIZE,
                                   per_device_eval_batch_size=BATCH_SIZE,
                                   num_train_epochs=train_epochs,
                                   fp16=True)
    base_trainer = Trainer(base_model, 
                           train_args,
                           train_dataset=dataset.train(),
                           eval_dataset=dataset.test(),
                           data_collator=data_collator,
                           tokenizer=tokenizer,
                           compute_metrics=_compute_accuracy)
    base_trainer.train()
    base_trainer.save_model(f'{save_dir}/base')

    # Extract and save evaluation metrics
    eval_metrics = [log for log in base_trainer.state.log_history if 'eval_accuracy' in log]
    for epoch, metrics in enumerate(eval_metrics, 1):
        base_trainer.save_metrics(f'eval_epoch_{epoch}', metrics)

    # Create and train smaller model
    print(f'{HOT_PINK}Training smaller model...{RESET}')
    smaller_model = deepcopy(base_model)
    reinitialize_weights(smaller_model)

    smaller_model = _make_pruned_model(
        RandUnstructured, smaller_model, prune_ptg)
    train_args = TrainingArguments(f'{save_dir}/smaller', 
                                   eval_strategy='epoch',
                                   save_strategy='no',
                                   logging_strategy='epoch',
                                   per_device_train_batch_size=BATCH_SIZE,
                                   per_device_eval_batch_size=BATCH_SIZE,
                                   num_train_epochs=train_epochs,
                                   fp16=True)
    smaller_trainer = Trainer(smaller_model, 
                           train_args,
                           train_dataset=dataset.train(),
                           eval_dataset=dataset.test(),
                           data_collator=data_collator,
                           tokenizer=tokenizer,
                           compute_metrics=_compute_accuracy)
    smaller_trainer.train()
    smaller_trainer.save_model(f'{save_dir}/smaller')

    # Extract and save evaluation metrics
    eval_metrics = [log for log in smaller_trainer.state.log_history if 'eval_accuracy' in log]
    for epoch, metrics in enumerate(eval_metrics, 1):
        smaller_trainer.save_metrics(f'eval_epoch_{epoch}', metrics)

    # Create and train pruned models 
    print(f'{HOT_PINK}Fine-tuning pruned models...{RESET}')
    base_model = AutoModelForSequenceClassification.from_pretrained(
        f'{save_dir}/base') # Prune from trained base model
    for method in pruning_methods:
        method_name = method.__name__
        pruned_model = _make_pruned_model(method, base_model, prune_ptg)
        train_args = TrainingArguments(f'{save_dir}/{method_name}', 
                                       eval_strategy='epoch',
                                       save_strategy='no',
                                       logging_strategy='epoch',
                                       per_device_train_batch_size=BATCH_SIZE,
                                       per_device_eval_batch_size=BATCH_SIZE,
                                       num_train_epochs=1,
                                       fp16=True)
        prune_trainer = Trainer(pruned_model, 
                                train_args,
                                train_dataset=dataset.train(),
                                eval_dataset=dataset.test(),
                                data_collator=data_collator,
                                tokenizer=tokenizer,
                                compute_metrics=_compute_accuracy)
        prune_trainer.train()
        prune_trainer.save_model(f'{save_dir}/{method_name}')

        # Extract and save evaluation metrics
        eval_metrics = [log for log in prune_trainer.state.log_history if 'eval_accuracy' in log]
        for epoch, metrics in enumerate(eval_metrics, 1):
            prune_trainer.save_metrics(f'eval_epoch_{epoch}', metrics)

    # Give user save locations
    print(f'{HOT_PINK}Models generated and saved to {save_dir}{RESET}')
    
def _compute_accuracy(preds):
    metric = evaluate.load('accuracy')
    logits, labels = preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def _make_pruned_model(method, base_model, prune_ptg):
    pruner = method()
    pruned_model = deepcopy(base_model)
    pruner.prune(get_params_to_prune(pruned_model), prune_ptg)
    return pruned_model

