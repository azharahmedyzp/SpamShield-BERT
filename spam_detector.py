import os
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {device}")

# Check if we have our own finetuned model, otherwise use a pre-trained fallback.
MODEL_PATH = "./finetuned-spam-model"
if os.path.exists(MODEL_PATH):
    print("Loading locally finetuned model...")
    model_id = MODEL_PATH
else:
    print("Local finetuned model not found. Using fallback model...")
    model_id = "mrm8488/bert-tiny-finetuned-sms-spam-detection"

spam_pipeline = pipeline(
    "text-classification",
    model=model_id,
    device=device
)

def detect_spam(email_text):
  result = spam_pipeline(email_text)[0]
  label_str = result["label"].upper()
  if "LABEL_1" in label_str or "SPAM" in label_str:
      label = "SPAM"
  else:
      label = "non-SPAM"
      
  confidence = round(result["score"] * 100, 2)
  return {
      "label": label,
      "confidence": confidence
  }