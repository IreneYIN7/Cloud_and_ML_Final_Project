from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Environment variable for SageMaker endpoint
SAGEMAKER_API_URL = os.getenv("SAGEMAKER_API_URL", "https://0kqaz13h1m.execute-api.us-east-1.amazonaws.com/default/sageMakerEndPointHandler")
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the frontend

# Mocked API endpoint
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    # try:
    #     response = requests.post(SAGEMAKER_API_URL, json={"text": text})
    #     response_data = response.json()
    #     return jsonify(response_data)
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    # Mock response instead of calling SageMaker
    mocked_response = {
        "generated_text": f"Mocked response for input: {text}"
    }
    return jsonify(mocked_response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


