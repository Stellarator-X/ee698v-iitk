import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D
import utils
import numpy as np
import os
import glob
import pandas as pd

# Global Variables - Required file/folder paths
MODEL_PATH = ""  # Add model path here
TASK1_PATH = ""  # Add task1 test cases folder path here
TASK1_SUBMISSION_PATH = ""  # Add the task 1 submission csv path here
TASK2_PATH = ""  # Add task 2 test cases folder path here
TASK2_SUBMISSION_PATH = ""  # Add the task 2 submission csv path here

# (Pre)Processing Functions
def pad_spectrogram(spec, pad_shape=(513, 400)):
    x, y = pad_shape
    if spec.shape == pad_shape:
        return spec
    padded = np.zeros((x, y))
    spec = spec[:x, :y]
    padded[: spec.shape[0], : spec.shape[1]] = spec
    return padded


def split_pad_spec(spec, time_interval=400):
    dt = time_interval
    n = int(spec.shape[1] / dt) + 1
    specs = []
    for i in range(n):
        specs.append(pad_spectrogram(spec[:, i * dt : (i + 1) * dt])[:, :, np.newaxis])
    return specs


def predict_sequence(instance_model, spec, n_beams=4):
    n_beams = 4
    beams = []
    for i in range(n_beams):
        pre, post = np.zeros((513, 100 * n)), np.zeros((513, 100 * (n_beams - n)))
        temp = np.append(np.append(pre, spec, axis=-1), post, axis=-1)
        beams.append(split_pad_spec(temp))
    preds = []
    for beam in beams:
        preds.append(instance_model.predict(beam))
    preds = np.array(preds)

    mean_preds = np.mean(preds, axis=0)
    return mean_preds


# Variables.
event_model = load_model(MODEL_PATH)
label2idx = {
    "air_conditioner": 0,
    "car_horn": 1,
    "children_playing": 2,
    "dog_bark": 3,
    "drilling": 4,
    "engine_idling": 5,
    "gun_shot": 6,
    "jackhammer": 7,
    "siren": 8,
    "street_music": 9,
}
idx2label = {label2idx[label]: label for label in label2idx}

# Prediction Pipeline for Task 1
## Reading the Test Cases
task1_files = glob.glob(TASK1_PATH + "/*.npy")
x_test = []
for file in task1_files:
    spec = np.load(file)
    # Padding the file
    x_test.append(pad_spectrogram(spec)[:, :, np.newaxis])
x_test = np.array(x_test)

## Getting the Predictions
y_pred = event_model.predict(x_test)
predictions = []
for pred in y_pred:
    idx = np.argmax(pred)
    label = idx2label[idx]
    predictions.append(label)
predictions = np.array(predictions)

##  Saving the Predicted Labels in the required format
submission = {"file": task1_files, "labels": predictions}
submission_frame = pd.DataFrame(submission)
submission_frame.to_csv(TASK1_SUBMISSION_PATH, header=False, index=False)

# TASK 2
## Prepping the Data
task2_files = glob.glob(TASK2_PATH + "/*.npy")
X_test = []
for file in task2_files:
    spec = np.load(file)
    X_test.append(spec)

## Getting the predictions
predictions2 = []
for instance in X_test:
    preds = predict_sequence(event_model, instance)
    preds = np.argmax(preds, axis=-1)
    labels = [idx2label[idx] for idx in preds]
    prediction = []
    prev = ""
    for label in labels:
        if label is not prev:
            prediction.append(label)
            prev = label
    predictions2.append("-".join(prediction))

## Saving the Predicted Labels in the required format.
submission2 = {"file": task2_files, "labels": predictions2}
submission2_frame = pd.DataFrame(submission2)
submission2_frame.to_csv(TASK2_SUBMISSION_PATH, header=False, index=False)
