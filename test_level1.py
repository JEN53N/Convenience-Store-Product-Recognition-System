import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ====== CONFIG ======
MODEL_PATH = "model_level1_mobilenet.h5"
CLASS_NAMES = ["food_and_beverages", "personal_care"]
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.9  # 70% confidence minimum

# ====== LOAD MODEL ======
model = load_model(MODEL_PATH)
print("✅ Level 1 Model loaded successfully.")

# ====== CAMERA ======
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict
    preds = model.predict(img, verbose=0)
    label_idx = np.argmax(preds)
    label = CLASS_NAMES[label_idx]
    confidence = preds[0][label_idx]

    # Show prediction only if confidence is strong enough
    if confidence >= CONFIDENCE_THRESHOLD:
        text = f"{label} ({confidence*100:.1f}%)"
        color = (0, 255, 0)
    else:
        text = "No product detected"
        color = (0, 0, 255)

    # Display on frame
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Level 1 Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#cv2.destroyAllWindows()
