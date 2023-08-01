import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "test")


def load_data(image_dir):
    images = []
    labels = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("JPG") or file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                gray = cv2.imread(path, 0)
                pic = cv2.adaptiveThreshold(
                    gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                bw = pic.reshape(-1)
                images.append(bw)
                labels.append(1 if "face" in root else -1)
    return images, labels


def preprocess_data(images, labels):
    scaler = MinMaxScaler()
    images = scaler.fit_transform(images)
    labels = np.array(labels)
    return images, labels


def build_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        100, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)
    return model


def predict_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).flatten().astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC:", roc_auc)
    return y_pred
