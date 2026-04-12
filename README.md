# Mail Spam Detector 📧

A real-time email spam detection web app powered by BERT and Flask.

## What it does
Paste any email — it instantly classifies it as **SPAM** or **HAM (Legitimate)** with a confidence score.

## Tech Stack
- 🤗 Hugging Face Transformers (BERT)
- 🐍 Python & Flask
- 🎨 HTML / CSS / JavaScript

## Model
The project uses a fine-tuned BERT model on the SMS Spam Collection dataset.
(Base model example: `prajjwal1/bert-tiny`)

## Project Structure
mail-spam-detector/
├── spam_detector.py   # ML model logic
├── finetune.py        # Model fine-tuning script
├── app.py             # Flask routes
├── requirements.txt   # Dependencies
└── templates/
    └── index.html     # Frontend UI

## What I Learned
- Finetuning transformer models with Hugging Face datasets and Trainer
- Using pretrained transformer models via pipeline()
- GPU vs CPU inference comparison
- Connecting an ML model to a Flask web app

## Author
Azhar Ahmed