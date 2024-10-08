import logging
from flask import Flask, request, jsonify
from transformers import ViTImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
from datetime import datetime
import socket

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='api_requests.log', 
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Load the processor and model outside of the route to avoid reloading it with each request

# processor = ViTImageProcessor.from_pretrained('yeftakun/vit-base-nsfw-detector')
# model = AutoModelForImageClassification.from_pretrained('yeftakun/vit-base-nsfw-detector')
processor = ViTImageProcessor.from_pretrained('../')
model = AutoModelForImageClassification.from_pretrained('../')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get the image URL from the POST request
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            error_msg = "Image URL not provided"
            logging.error(f"{error_msg} | IP: {request.remote_addr}")
            return jsonify({"error": error_msg}), 400

        # Fetch the image from the URL
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Convert the image to RGB mode if it's not already in RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Run the image through the model
        outputs = model(**inputs)
        logits = outputs.logits

        # Get the predicted class
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        # Log the successful request
        logging.info(f"Classified image | IP: {request.remote_addr} | URL: {image_url} | Class: {predicted_class}")

        # Return the classification result
        return jsonify({
            "image_url": image_url,
            "predicted_class": predicted_class
        })
    
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing request: {error_msg} | IP: {request.remote_addr}")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)
