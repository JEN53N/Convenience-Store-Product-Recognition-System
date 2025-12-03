import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# ============================
# CONFIGURATION
# ============================
DATASET_DIR = "dataset_level1"   # Make sure this path is correct
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15                    # You can increase later if needed
MODEL_NAME = "model_level1_mobilenet.h5"

# ============================
# DATA PREPARATION
# ============================
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# ============================
# MODEL CREATION
# ============================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Freeze base model for faster training

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================
# TRAINING
# ============================
print("\n🚀 Starting Level-1 Training: Food & Beverages vs Personal Care\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ============================
# SAVE MODEL
# ============================
model.save(MODEL_NAME)
print(f"\n✅ Level-1 model saved as {MODEL_NAME}")

# ============================
# CLASS LABELS (for prediction stage)
# ============================
print("\n🔖 Class Indices:")
print(train_data.class_indices)
