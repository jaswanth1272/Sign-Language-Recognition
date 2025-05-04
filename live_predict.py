import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('sign_model.h5')
label_map = np.load('label_map.npy', allow_pickle=True).item()
rev_label_map = {v: k for k, v in label_map.items()}

IMG_SIZE = 64

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    pred = model.predict(reshaped, verbose=0)
    pred_idx = np.argmax(pred)
    pred_label = rev_label_map.get(pred_idx, "?")

    cv2.putText(frame, f'Prediction: {pred_label}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv2.imshow("Real-Time Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
