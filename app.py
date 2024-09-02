import cv2
import numpy as np
from flask import Flask, request, jsonify

import image_processing

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_image():
    # Read the image from the request
    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Process the image to determine if the plant needs watering
    needs_watering = image_processing.predict_image(img)

    if needs_watering:
        return jsonify({"command": "water"})
    else:
        return jsonify({"command": "do_nothing"})


if __name__ == "__main__":
    app.run()
