# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:00:25 2024

@author: SyHAF
"""

import cv2
import numpy as np


# Function to process the image using the loaded CNN model
# def process_image(img, model, input_shape=(64, 64)):
#     # Resize the image to match the input shape of the CNN
#     img_resized = cv2.resize(img, input_shape)
#
#     # Normalize the image data
#     img_normalized = img_resized / 255.0
#
#     # Add batch dimension (since the model expects batches of images)
#     img_batch = np.expand_dims(img_normalized, axis=0)
#
#     # Predict the green probability
#     green_prob = model.predict(img_batch)[0][0]
#
#     # If the green probability is less than 50%, assume the plant needs watering
#     return green_prob < 0.5

def predict_image(img, model):
    # Load the image
    # img = cv2.imread(img_path)

    # Resize the image to match the input shape of the CNN (e.g., 64x64)
    input_shape = (224, 224)
    img_resized = cv2.resize(img, input_shape)

    # Normalize the image data to be in the range [0, 1]
    img_normalized = img_resized / 255.0

    # Expand dimensions to match the input shape of the model (batch size, height, width, channels)
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Predict using the loaded model
    prediction = model.predict(img_expanded)

    # Convert the prediction result into a meaningful label (e.g., healthy/unhealthy)
    if prediction[0][0] < 0.5:
        output = True
    else:
        output = False

    return output
