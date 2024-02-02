from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import requests
from flask_cors import cross_origin

app = Flask(__name__)

# Load the pre-trained model
modelo = tf.keras.models.load_model("detectar_picadura.h5", custom_objects={'KerasLayer': hub.KerasLayer})

def categorize(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img).astype(float) / 255

    img = cv2.resize(img, (224, 224))
    prediction = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediction[0], axis=-1)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.get_json()
        image_url = data.get('image_url')

        if image_url:
            prediction = categorize(image_url)

            # Map predictions to corresponding classes
            classes = ["Abeja", "Alergia", "Araña", "Garrapata", "Mosquito"]
            insect = classes[prediction]

            return jsonify({"insect": insect})
        else:
            return jsonify({"error": "Missing 'image_url' parameter"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()