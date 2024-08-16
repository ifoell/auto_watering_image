# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:55:22 2024

@author: SyHAF
"""

import platform
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
    # Create and/or load the CNN model
    input_shape = (64, 64, 3)
    model = image_processing.create_cnn_model(input_shape)

    # Process the image to determine if the plant needs watering
    needs_watering = image_processing.process_image(img, model, input_shape[:2])

    if needs_watering:
        return jsonify({"command": "water"})
    else:
        return jsonify({"command": "do_nothing"})


if __name__ == "__main__":
    # Check the System Type before to decide to bind
    # If the system is a Linux machine -:)
    if platform.system() == "Linux":
        app.run(host='0.0.0.0', port=5000, debug=True)
    # If the system is a windows /!\ Change  /!\ the   /!\ Port
    elif platform.system() == "Windows":
        app.run(host='0.0.0.0', port=50000, debug=True)
