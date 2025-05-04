# prepare_data.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_DIR = 'dataset/Gesture Image Data'
IMG_SIZE = 64
labels = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

data = []
target = []

for label in labels:
    path = os.path.join(DATA_DIR, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        target.append(label_map[label])

data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
target = to_categorical(target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

np.savez('data_ready.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, label_map=label_map)
