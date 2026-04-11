
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {device}")

spam_pipeline = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    device=device
)

def detect_spam(email_text):
  result = spam_pipeline(email_text)[0]
  label = "SPAM" if result["label"] == "LABEL_1" else "non-SPAM"
  confidence = round(result["score"] * 100, 2)
  return {
      "label": label,
      "confidence": confidence
  }
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {device}")

spam_pipeline = pipeline(
    "text-classification",
    model="mshenoda/roberta-spam",
    device=device
)

def detect_spam(email_text):
  result = spam_pipeline(email_text)[0]
  label = "SPAM" if result["label"] == "LABEL_1" else "non-SPAM"
  confidence = round(result["score"] * 100, 2)
  return {
      "label": label,
      "confidence": confidence
 
  }