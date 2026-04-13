import os
import re
import torch
from transformers import pipeline

# ──────────────────────────────────────────────────────────────────────────────
# THE NEURAL ENGINE
# This is where we load the custom intelligence you trained!
# ──────────────────────────────────────────────────────────────────────────────

# Automatically detecting if we have a GPU available for extreme speed!
device = 0 if torch.cuda.is_available() else -1

# We prioritize your custom-trained 'SpamShield' model for maximum accuracy.
MODEL_PATH = "./finetuned-spam-model"

if os.path.exists(MODEL_PATH):
    model_id = MODEL_PATH
    print(f"✨ Custom Neural Engine detected! Initializing SpamShield...")
else:
    # If a custom model isn't trained yet, we use a robust DistilBERT fallback.
    model_id = "mshenoda/roberta-spam"
    print(f"💡 Initializing high-speed fallback engine ({model_id})...")

# Building the analysis pipeline
print("🚀 Engine ready. Preparing to scan for threats.")
spam_pipeline = pipeline("text-classification", model=model_id, device=device)

# ──────────────────────────────────────────────────────────────────────────────
# INTELLIGENT PREPROCESSING
# Cleaning the data ensures our model isn't distracted by messy formatting.
# ──────────────────────────────────────────────────────────────────────────────
def clean_payload(text):
    # We remove excessive spaces and standardize the text for more reliable results.
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def detect_spam(email_text):
    """
    Analyzes an email to verify its safety. 
    Returns a dictionary with classification and confidence.
    """
    # Clean the data first for 'perfect' results.
    cleaned = clean_payload(email_text)
    
    # We analyze up to 2500 characters to capture the full semantic context.
    result = spam_pipeline(cleaned[:2500])[0]
    
    # Normalizing the engine's output for our beautiful UI.
    label_raw = result["label"].upper()
    is_spam = "SPAM" in label_raw or "LABEL_1" in label_raw
    
    label = "SPAM" if is_spam else "non-SPAM"
    confidence = round(result["score"] * 100, 2)
    
    return {
        "label": label, 
        "confidence": confidence
    }
