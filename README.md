# Mail Spam Detector 📧

A real-time email spam detection web app powered by BERT and Flask.

## What it does
Paste any email — it instantly classifies it as **SPAM** or **HAM (Legitimate)** with a confidence score.

## Tech Stack
- 🤗 Hugging Face Transformers (BERT)
- 🐍 Python & Flask
- 🎨 HTML / CSS / JavaScript

## Model
`mrm8488/bert-tiny-finetuned-sms-spam-detection`  
Pretrained BERT model fine-tuned on the SMS Spam Collection dataset.


## Project Structure
mail-spam-detector/
├── spam_detector.py   # ML model logic
├── app.py             # Flask routes
├── requirements.txt   # Dependencies
└── templates/
    └── index.html     # Frontend UI

## What I Learned
- Using pretrained transformer models via pipeline()
- GPU vs CPU inference comparison
- Batch inference for better throughput
- Connecting an ML model to a Flask web app

## Author
Azhar Ahmed