import os
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    print("Loading dataset...")
    # Load the SMS Spam Collection dataset
    dataset = load_dataset("sms_spam")
    
    # The dataset has 'sms' and 'label' columns. We need to rename 'sms' to 'text' for standard processing,
    # or just use 'sms' during tokenization.
    
    model_name = "prajjwal1/bert-tiny"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["sms"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Train/test split is not explicitly in sms_spam (it usually has just 'train'),
    # so we split the 'train' split.
    if 'test' not in tokenized_datasets.keys():
        tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)

    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # We should set the label mappings so the inference pipeline correctly shows SPAM / non-SPAM
    model.config.id2label = {0: "non-SPAM", 1: "SPAM"}
    model.config.label2id = {"non-SPAM": 0, "SPAM": 1}

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting finetuning...")
    trainer.train()

    print("Saving finetuned model...")
    save_path = "./finetuned-spam-model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}. You can now run the Flask app!")

if __name__ == "__main__":
    main()
