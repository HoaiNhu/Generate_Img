from flask import Flask, request, jsonify
from generate_cake import process_sketch
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is running"})

@app.route("/generate-cake", methods=["POST"])
def generate_cake():
    data = request.get_json()
    sketch = data.get("sketch")
    if not sketch:
        return jsonify({"error": "No sketch provided"}), 400

    try:
        result_image = process_sketch(sketch)
        if isinstance(result_image, str) and "error" in result_image.lower():
            return jsonify({"error": result_image}), 500
        return jsonify({"image": result_image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)