"""
    The fine tuning techniques might matter.
    Should we therefore use a range of different approaches?
"""

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments




# Fine-tune the pruned BERT model
def Bert_fine_tuner(model):
    # Load the Yelp polarity dataset
    dataset = load_dataset("yelp_polarity", split="train[:10%]")
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer method for encoding the dataset
    def Bert_tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    encoded_dataset = dataset.map(Bert_tokenize_function, batched=True)

    
    # Define the TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",           # Directory to save checkpoints
        eval_strategy="epoch",            # Evaluate at the end of each epoch
        learning_rate=2e-5,               # Learning rate
        per_device_train_batch_size=16,   # Batch size for training          
        per_device_eval_batch_size=16,    # Batch size for evaluation
        num_train_epochs=3,               # Number of training epochs
        weight_decay=0.01,                # Weight decay for regularization
        logging_dir="./logs",             # Directory to save logs
        logging_steps=10,                 # Log every 10 steps
        load_best_model_at_end=True,      # Optional: Load the best model after training
        save_total_limit=2,               # Keep only the last 2 checkpoints
        save_strategy="epoch",            # Save a checkpoint at the end of each epoch
    )

    #Define the Trainer
    trainer = Trainer(
        model=model,                      # The BERT model
        args=training_args,               # Training arguments
        train_dataset=encoded_dataset,    # Training dataset
        eval_dataset=encoded_dataset,     # Evaluation dataset (can be separate)
    )

    # fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

    print("Model fine-tuned and saved successfully!")

    # Evaluate the fine-tuned model
    eval_result = trainer.evaluate()
    print(f"Evaluation Results: {eval_result}")