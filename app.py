from flask import Flask, request, jsonify
from transformers import ViTImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# Initialize the Flask app
app = Flask(__name__)

# Load the processor and model outside of the route to avoid reloading it with each request
processor = ViTImageProcessor.from_pretrained('yeftakun/vit-base-nsfw-detector')
model = AutoModelForImageClassification.from_pretrained('yeftakun/vit-base-nsfw-detector')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get the image URL from the POST request
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({"error": "Image URL not provided"}), 400

        # Fetch the image from the URL
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Run the image through the model
        outputs = model(**inputs)
        logits = outputs.logits

        # Get the predicted class
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        # Return the classification result
        return jsonify({
            "image_url": image_url,
            "predicted_class": predicted_class
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
