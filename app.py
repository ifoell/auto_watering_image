import logging

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

import image_processing

app = Flask(__name__)

# Ensure the logging directory exists
log_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory of the script
log_file = os.path.join(log_dir, 'app.log')  # Path to the log file
print(f"Logging to: {log_file}")

# Setup logging
try:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Force to override any previous logging configuration
    )

    # Also log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(console_handler)

    logging.info("Logging setup complete.")
except Exception as e:
    print(f"Failed to configure logging: {e}")


# @app.route('/upload', methods=['POST'])
# def upload_image():
#     # Read the image from the request
#     file = request.files['file']
#     npimg = np.frombuffer(file.read(), np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#
#     # Process the image to determine if the plant needs watering
#     needs_watering = image_processing.predict_image(img)
#
#     if needs_watering:
#         print("command : water")
#         return jsonify({"command": "water"})
#     else:
#         print("command : do_nothing")
#         return jsonify({"command": "do_nothing"})
#

@app.route('/upload', methods=['POST'])
def upload_image():
    # Log that the route was accessed
    # logging.info("Received image upload request.")
    #

    # Log that the route was accessed
    logging.info("Received image upload request.")

    if request.data == b'':
        # this is to test from postman
        # Check if the request has a file
        if 'data' not in request.files:
            logging.error("No image part found in request.")
            return jsonify({"error": "No image part in request"}), 400

        # Retrieve the image file
        file = request.files['data']

        # Log the file content type
        logging.info(f"Content-Type received: {file.content_type}")

        # Check if the file is an image
        if file.content_type not in ['image/jpeg', 'image/png']:
            logging.error("Invalid file type. Expected an image format.")
            return jsonify({"error": "File must be an image format (e.g., image/jpeg, image/png)"}), 400

        # Read the image data using OpenCV
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    else:
        # this is from esp32-cam
        # Check if the content type is image
        if 'image' not in request.content_type:
            logging.error("Invalid Content-Type. Expected an image format.")
            return jsonify({"error": "Content-Type must be an image format (e.g., image/jpeg, image/png)"}), 400

        # Log the content type
        logging.info(f"Content-Type received: {request.content_type}")

        # Read the raw image data from the request
        npimg = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Log that the image was successfully decoded
    if img is not None:
        logging.info("Image successfully decoded.")
    else:
        logging.error("Failed to decode the image.")
        return jsonify({"error": "Failed to decode the image"}), 400

    # Process the image to determine if the plant needs watering
    try:
        needs_watering = image_processing.predict_image(img)
        logging.info("Image processing completed.")
    except Exception as ex:
        logging.error(f"Image processing failed: {ex}")
        return jsonify({"error": "Image processing failed"}), 500

    # Log the decision and return the appropriate command
    if needs_watering:
        logging.info("Plant needs watering. Sending 'water' command.")
        print("command : water")
        return jsonify({"command": "water"})
    else:
        logging.info("Plant does not need watering. Sending 'do_nothing' command.")
        print("command : do_nothing")
        return jsonify({"command": "do_nothing"})


@app.route('/cek', methods=['GET'])
def cek():
    logging.info("Received cek request.")
    print("command : cek")
    return jsonify({"command": "cek"})


if __name__ == "__main__":
    app.run()
