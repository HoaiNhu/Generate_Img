from flask import Flask, request, jsonify
from generate_cake import process_sketch
import os
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    logger.info("Health check endpoint called")
    return jsonify({"status": "API is running"})

@app.route("/generate-cake", methods=["POST"])
def generate_cake():
    try:
        logger.info("Received generate-cake request")
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data in request")
            return jsonify({"error": "No JSON data provided"}), 400
            
        sketch = data.get("sketch")
        if not sketch:
            logger.error("No sketch provided in request")
            return jsonify({"error": "No sketch provided"}), 400

        logger.info("Processing sketch, length: %d", len(sketch))
        try:
            result_image = process_sketch(sketch)
            if isinstance(result_image, str) and "error" in result_image.lower():
                logger.error("Error in process_sketch: %s", result_image)
                return jsonify({"error": result_image}), 500
                
            logger.info("Successfully generated image, length: %d", len(result_image))
            return jsonify({"image": result_image})
            
        except Exception as e:
            logger.error("Error processing sketch: %s", str(e), exc_info=True)
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error("Unexpected error in generate-cake endpoint: %s", str(e), exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port)