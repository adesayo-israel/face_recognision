import math
import random
import string
import numpy as np
import cv2

from PIL import Image
import pickle
import gzip
import os
import time

start_time = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "test")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load(file_name):
    # load the model
    with gzip.open(file_name, "rb") as stream:
        model = pickle.load(stream)
    return model


def save(file_name, model):
    # save the model
    with gzip.open(file_name, "wb") as stream:
        pickle.dump(model, stream)


wi, wo, wt, w1, w2, b = load("BPN_new")
count = 0
letter = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("JPG") or file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            gray = cv2.imread(path, 0)
            pic = cv2.adaptiveThreshold(
                gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            bw1 = pic.reshape(-1)
            bw = np.where(bw1 == 0, -1, 1)
            bw = np.append(bw, 1.0)
            h = np.zeros(200)
            for j in range(200):
                h[j] = sigmoid(np.sum(bw * wi[:, j]))
            o1 = np.zeros(2)
            for k in range(2):
                o1[k] = np.sum(h * wt[k, :])
            su = b + (o1[0] * w1) + (o1[1] * w2)
            if su/500000 < 0.85:
                count += 1
            letter.append(su)

print("Mean:", np.mean(letter) / 500000)
print("Percentage:", ((len(letter) - count) / len(letter)) * 100)
print("--- %s seconds ---" % (time.time() - start_time))
