from flask import Flask, request, jsonify, render_template
from spam_detector import detect_spam

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email = data["email"]
    output = detect_spam(email)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
