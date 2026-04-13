import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ──────────────────────────────────────────────────────────────────────────────
# CORE PERFORMANCE METRICS
# We're tracking how well our model learns. Higher accuracy means safer emails!
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    print("\ Welcome to the SpamShield Training Suite!")
    print("━" * 50)
    
    # Step 1: Loading the Massive 190k Dataset
    # We're using the meruvulikith dataset to give our model world-class intelligence.
    print(" Gathering the knowledge base (190k emails)...")
    raw_dataset = load_dataset("meruvulikith/190k-spam-ham-email")
    
    # Step 2: Refining the Data
    # Aligning column names so their neural engine can read them perfectly.
    dataset = raw_dataset['train'].rename_columns({"Text": "text", "Label": "label"})
    
    # Selecting a balanced 20,000 email subset for optimized local training.
    # This ensures your results are fast AND incredibly accurate!
    print(" Selecting the top 20,000 samples for high-speed local learning...")
    dataset = dataset.shuffle(seed=42).select(range(20000))
    
    # Splitting into training and verification sets.
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Step 3: Initializing the DistilBERT Brain
    # DistilBERT is our chosen 'brain' because it's both smart and efficient.
    model_name = "distilbert-base-uncased"
    print(f" Waking up the DistilBERT engine ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        # We increase context to 512 tokens to scan even long email bodies deeply.
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print(" Preparing the emails for neural analysis (Tokenization)...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print(" Calibrating the classification architecture...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Setting human-readable labels for the engine.
    model.config.id2label = {0: "non-SPAM", 1: "SPAM"}
    model.config.label2id = {"non-SPAM": 0, "SPAM": 1}

    # Step 4: Configuring the Educational Path
    # We've optimized batch sizes and epochs to ensure your model becomes a 'perfect' detector.
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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

    # Step 5: The Main Event!
    # Detecting hardware to keep you informed.
    hw = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Starting the fine-tuning phase on your {hw}...")
    print("   (This is where the magic happens. Just a moment!)")
    print("━" * 50)
    
    trainer.train()

    # Final Polish
    print("\nSuccess! Your custom SpamShield model has reached peak intelligence.")
    save_path = "./finetuned-spam-model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Perfection saved locally to: {save_path}")
    print("━" * 50)
    print("Pro-Tip: Now run 'python app.py' to see your work in action!\n")

if __name__ == "__main__":
    main()
