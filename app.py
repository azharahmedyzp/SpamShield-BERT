from flask import Flask, request, jsonify, render_template
from spam_detector import detect_spam

# ──────────────────────────────────────────────────────────────────────────────
# SPAMSHIELD WEB COMMAND CENTER
# This server handles all the requests from your elite dashboard.
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

@app.route("/")
def home():
    """Welcome to the main dashboard."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives an email message, cleans it, and runs it through the neural engine.
    Returns the real-time safety classification.
    """
    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        
        if not email:
            return jsonify({"error": "No message provided"}), 400
            
        # Passing the message to our high-performance detection logic.
        output = detect_spam(email)
        return jsonify(output)
        
    except Exception as e:
        # Gracefully handling any unexpected issues to keep the app running smooth.
        print(f"⚠️ App Notice: {str(e)}")
        return jsonify({"error": "Neural engine is busy. Please try again."}), 500

if __name__ == "__main__":
    print("\n🚀 SpamShield Landing Site is active!")
    print("✨ View your dashboard at: http://127.0.0.1:5000")
    print("━" * 50)
    app.run(debug=True)