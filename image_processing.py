# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:00:25 2024

@author: SyHAF
"""

import cv2
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.api.optimizers import Adam


def create_cnn_model(input_shape):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))

    # Output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Function to process the image using CNN
def process_image(img, model, input_shape=(64, 64)):
    # Resize the image to match the input shape of the CNN
    img_resized = cv2.resize(img, input_shape)

    # Normalize the image data
    img_normalized = img_resized / 255.0

    # Predict the green ratio (assuming 1 for green, 0 for not green)
    green_prob = model.predict(np.expand_dims(img_normalized, axis=0))[0][0]

    # If the green probability is less than 50%, assume the plant needs watering
    if green_prob < 0.5:
        return True
    return False
