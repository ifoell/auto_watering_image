# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:00:25 2024

@author: SyHAF
"""

import cv2
import numpy as np
# from keras._tf_keras.keras.models import load_model
from keras.models import load_model

# Load the trained CNN model
model = load_model('sawi_model_tf_2161.h5')


def predict_image(img):
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
    if prediction[0][0] < 0.2:
        output = True
    else:
        output = False

    return output
