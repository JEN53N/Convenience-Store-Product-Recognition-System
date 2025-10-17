import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# 1️⃣ Load the trained model
# -------------------------------
model = load_model("product_identifier_mobilenet.h5")

# -------------------------------
# 2️⃣ Load class labels from training folder
# -------------------------------
train_dir = "dataset/train"  # Path to your dataset
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=(224,224),   # Must match MobileNetV2 input
    batch_size=32
)
labels = list(train_gen.class_indices.keys())  # ['Chips_Lays_Classic', 'Biscuits_ParleG', ...]

# -------------------------------
# 3️⃣ Start webcam
# -------------------------------
cap = cv2.VideoCapture(0)
confidence_threshold = 0.6  # 60%

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # Preprocess the frame
    # -------------------------------
    img = cv2.resize(frame, (224,224))
    img = np.expand_dims(img/255.0, axis=0)

    # -------------------------------
    # Make prediction
    # -------------------------------
    pred = model.predict(img)
    label = labels[np.argmax(pred)]
    confidence = np.max(pred)

    # -------------------------------
    # Only show prediction if confident
    # -------------------------------
    if confidence >= confidence_threshold:
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "No product detected", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Show frame
    cv2.imshow("Product Identifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#cv2.destroyAllWindows()


