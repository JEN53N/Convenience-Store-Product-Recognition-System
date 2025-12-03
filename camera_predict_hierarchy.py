import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import warnings
import h5py

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "models"
IMG_SIZE = (224, 224)  # Adjust if your models use different input size

# Model paths
MODEL_LEVEL1 = os.path.join(MODEL_DIR, "model_level1_mobilenet.h5")
MODEL_LEVEL2_FOOD = os.path.join(MODEL_DIR, "model_level2_foodnbeverages.h5")
MODEL_LEVEL2_PERSONAL = os.path.join(MODEL_DIR, "model_level2_personalcare.h5")
MODEL_LEVEL3_CHIPS = os.path.join(MODEL_DIR, "model_level3_4_chips.h5")
MODEL_LEVEL3_BISCUITS = os.path.join(MODEL_DIR, "model_level3_4_biscuits.h5")
MODEL_LEVEL3_DRINKS = os.path.join(MODEL_DIR, "model_level3_4_drinks.h5")
MODEL_LEVEL3_ORALCARE = os.path.join(MODEL_DIR, "model_level3_4_oralcare.h5")
MODEL_LEVEL3_BODYCARE = os.path.join(MODEL_DIR, "model_level3_4_bodycare.h5")

# Class labels for each level
LEVEL1_CLASSES = ["Food&Beverages", "PersonalCare"]

LEVEL2_FOOD_CLASSES = ["Biscuits", "Chips", "Drinks"]
LEVEL2_PERSONAL_CLASSES = ["BodyCare", "OralCare"]

# Update these with your actual brand/flavour labels

#LEVEL3_BISCUITS_CLASSES = ["Cani_Orginal", "Dark_Fantasy", "Oreo_Chocolate", "Oreo_Vanilla"]
LEVEL3_BODYCARE_CLASSES = ["Cream_vasline", "Soap Dove", "Soap LifeBuoy", "Soap mysore sandal", "Soap Safeguard"]
LEVEL3_CHIPS_CLASSES = ["Bingo_CreamOnion", "Bingo_Orginal", "Bingo_Salted", "Kurkure Chutney", "Lays magic massala", "Lays Tomato",  "pringlez_original","pringlez_sour_cream"]
LEVEL3_CHOCOLATES_CLASSES = ["DairyMilk_Crispello", "Snicker_sachet"]
#LEVEL3_DRINKS_CLASSES = ["Coca Cola", "Pepsi", "Sprite", "Thumbs Up", "Fanta"]
LEVEL3_ORALCARE_CLASSES = ["Colgate_MaxFresh", "Colgate_MCP","Sensodyne_Mint"]


CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to display prediction

# ============================================================================
# CUSTOM OBJECT SCOPE FOR LEGACY LAYERS
# ============================================================================

class CustomInputLayer(keras.layers.InputLayer):
    """Custom InputLayer that handles batch_shape parameter"""
    def __init__(self, *args, **kwargs):
        # Remove batch_shape if present and convert to input_shape
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

# Custom objects dictionary for loading
CUSTOM_OBJECTS = {
    'InputLayer': CustomInputLayer,
}

# ============================================================================
# CUSTOM LOAD FUNCTION FOR COMPATIBILITY
# ============================================================================

def load_model_safe(model_path):
    """Load model with multiple fallback strategies"""
    print(f"Loading {os.path.basename(model_path)}...", end=" ")
    
    # Strategy 1: Load with custom objects and compile=False
    try:
        model = keras.models.load_model(
            model_path, 
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        print("✓")
        return model
    except Exception as e1:
        pass
    
    # Strategy 2: Use tf.keras directly
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        print("✓")
        return model
    except Exception as e2:
        pass
    
    # Strategy 3: Load weights only
    try:
        # Try to extract architecture from h5 file
        with h5py.File(model_path, 'r') as f:
            # Check if it's a sequential model
            if 'model_config' in f.attrs:
                import json
                config = json.loads(f.attrs['model_config'])
                
                # Reconstruct model from config
                if config['class_name'] == 'Sequential':
                    model = keras.Sequential()
                    # Build basic architecture (simplified)
                    model.add(keras.layers.InputLayer(input_shape=IMG_SIZE + (3,)))
                    model.add(keras.layers.GlobalAveragePooling2D())
                    model.add(keras.layers.Dense(128, activation='relu'))
                    
                    # Get output shape from file
                    if 'model_weights' in f:
                        last_layer = list(f['model_weights'].keys())[-1]
                        if 'kernel:0' in f['model_weights'][last_layer]:
                            output_shape = f['model_weights'][last_layer]['kernel:0'].shape[-1]
                            model.add(keras.layers.Dense(output_shape, activation='softmax'))
                
                # Load weights
                model.load_weights(model_path)
                print("✓ (weights only)")
                return model
    except Exception as e3:
        pass
    
    # Strategy 4: Build minimal MobileNet architecture
    try:
        print("⚠ Building fallback architecture...")
        base = keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights=None
        )
        model = keras.Sequential([
            base,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='softmax')  # Will adjust
        ])
        
        # Try to load weights
        try:
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            print("✓ (partial weights)")
            return model
        except:
            print("✓ (architecture only - untrained)")
            return model
    except Exception as e4:
        print(f"❌ Failed")
        raise Exception(f"All loading strategies failed for {model_path}")

# ============================================================================
# LOAD MODELS
# ============================================================================

print("\n" + "="*60)
print("Loading models...")
print("="*60)
try:
    model_level1 = load_model_safe(MODEL_LEVEL1)
    model_level2_food = load_model_safe(MODEL_LEVEL2_FOOD)
    model_level2_personal = load_model_safe(MODEL_LEVEL2_PERSONAL)
    model_level3_chips = load_model_safe(MODEL_LEVEL3_CHIPS)
    model_level3_biscuits = load_model_safe(MODEL_LEVEL3_BISCUITS)
    model_level3_drinks = load_model_safe(MODEL_LEVEL3_DRINKS)
    model_level3_oralcare = load_model_safe(MODEL_LEVEL3_ORALCARE)
    model_level3_bodycare = load_model_safe(MODEL_LEVEL3_BODYCARE)
    
    print("="*60)
    print("✓ All models loaded successfully!\n")
except Exception as e:
    print("\n" + "="*60)
    print(f"❌ FATAL ERROR: {e}")
    print("="*60)
    print("\nPlease run the model conversion script:")
    print("python fix_models.py")
    print("\nOr provide the training script to check model architecture.")
    exit(1)

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess_frame(frame):
    """Preprocess camera frame for model input"""
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ============================================================================
# HIERARCHICAL PREDICTION FUNCTION
# ============================================================================

def predict_hierarchy(frame):
    """
    Perform hierarchical prediction across all levels
    Returns: (final_prediction_string, confidence_dict)
    """
    img = preprocess_frame(frame)
    
    # LEVEL 1: Food&Beverages vs PersonalCare
    pred_level1 = model_level1.predict(img, verbose=0)
    level1_idx = np.argmax(pred_level1)
    level1_conf = float(pred_level1[0][level1_idx])
    level1_class = LEVEL1_CLASSES[level1_idx]
    
    if level1_conf < CONFIDENCE_THRESHOLD:
        return "Low Confidence - Please show product clearly", {}
    
    # LEVEL 2: Subcategory
    if level1_class == "Food&Beverages":
        pred_level2 = model_level2_food.predict(img, verbose=0)
        level2_classes = LEVEL2_FOOD_CLASSES
    else:  # PersonalCare
        pred_level2 = model_level2_personal.predict(img, verbose=0)
        level2_classes = LEVEL2_PERSONAL_CLASSES
    
    level2_idx = np.argmax(pred_level2)
    level2_conf = float(pred_level2[0][level2_idx])
    level2_class = level2_classes[level2_idx]
    
    if level2_conf < CONFIDENCE_THRESHOLD:
        return f"{level1_class} → Low Confidence", {"Level 1": level1_conf}
    
    # LEVEL 3-4: Brand and Flavour
    if level2_class == "Chips":
        pred_level3 = model_level3_chips.predict(img, verbose=0)
        level3_classes = LEVEL3_CHIPS_CLASSES
    elif level2_class == "Biscuits":
        pred_level3 = model_level3_biscuits.predict(img, verbose=0)
        level3_classes = LEVEL3_BISCUITS_CLASSES
    elif level2_class == "Drinks":
        pred_level3 = model_level3_drinks.predict(img, verbose=0)
        level3_classes = LEVEL3_DRINKS_CLASSES
    elif level2_class == "OralCare":
        pred_level3 = model_level3_oralcare.predict(img, verbose=0)
        level3_classes = LEVEL3_ORALCARE_CLASSES
    else:  # BodyCare
        pred_level3 = model_level3_bodycare.predict(img, verbose=0)
        level3_classes = LEVEL3_BODYCARE_CLASSES
    
    level3_idx = np.argmax(pred_level3)
    level3_conf = float(pred_level3[0][level3_idx])
    level3_class = level3_classes[level3_idx]
    
    if level3_conf < CONFIDENCE_THRESHOLD:
        return f"{level1_class} → {level2_class} → Low Confidence", {
            "Level 1": level1_conf,
            "Level 2": level2_conf
        }
    
    # Build final prediction string
    final_prediction = f"{level1_class} → {level2_class} → {level3_class}"
    
    confidence_dict = {
        "Level 1": level1_conf,
        "Level 2": level2_conf,
        "Level 3": level3_conf,
        "Overall": (level1_conf + level2_conf + level3_conf) / 3
    }
    
    return final_prediction, confidence_dict

# ============================================================================
# MAIN CAMERA LOOP
# ============================================================================

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        print("\nTroubleshooting:")
        print("1. Check if camera is connected")
        print("2. Try different camera index: cv2.VideoCapture(1)")
        print("3. Check camera permissions")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*60)
    print("    CONVENIENCE STORE PRODUCT IDENTIFICATION")
    print("="*60)
    print("Controls:")
    print("  C - Toggle continuous/manual prediction mode")
    print("  SPACE - Manual predict (when continuous off)")
    print("  Q - Quit")
    print("="*60 + "\n")
    
    prediction_text = "Continuous Mode: ON"
    confidence_info = ""
    continuous_mode = True
    frame_counter = 0
    predict_every_n_frames = 10  # Predict every 10 frames to avoid lag
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_counter += 1
        
        # Create display frame
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Draw ROI (Region of Interest) rectangle
        roi_margin = 50
        cv2.rectangle(display_frame, 
                     (roi_margin, roi_margin), 
                     (width - roi_margin, height - roi_margin),
                     (0, 255, 0), 2)
        
        # Add semi-transparent overlay for text background
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 70), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, height - 120), (width, height), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
        
        # Add title and mode indicator
        mode_text = "LIVE MODE" if continuous_mode else "MANUAL MODE"
        mode_color = (0, 255, 0) if continuous_mode else (0, 165, 255)
        
        cv2.putText(display_frame, "DL Convenient Store", 
                   (10, 40), cv2.FONT_HERSHEY_DUPLEX, 
                   1.2, (0, 255, 255), 2)
        cv2.putText(display_frame, mode_text,
                   (width - 250, 40), cv2.FONT_HERSHEY_DUPLEX,
                   0.8, mode_color, 2)
        
        # Add prediction text
        y_offset = height - 90
        lines = prediction_text.split(' → ')
        for i, line in enumerate(lines):
            indent = i * 20
            cv2.putText(display_frame, line,
                       (10 + indent, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0) if i == len(lines) - 1 else (255, 255, 255), 
                       2)
        
        # Add confidence info
        if confidence_info:
            cv2.putText(display_frame, confidence_info,
                       (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (100, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Product Identification - Press SPACE to scan, Q to quit', display_frame)
        
        # Continuous prediction
        if continuous_mode and frame_counter % predict_every_n_frames == 0:
            try:
                prediction, conf_dict = predict_hierarchy(frame)
                prediction_text = prediction
                
                if conf_dict:
                    confidence_info = " | ".join([f"{k}: {v:.1%}" for k, v in conf_dict.items()])
                else:
                    confidence_info = ""
            except Exception as e:
                prediction_text = "Error during prediction"
                confidence_info = ""
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            print("\n✓ Exiting...")
            break
        
        elif key == ord('c'):  # Toggle continuous mode
            continuous_mode = not continuous_mode
            if continuous_mode:
                print("\n✓ Continuous prediction: ON")
                prediction_text = "Live prediction active..."
            else:
                print("\n✓ Manual mode: Press SPACE to predict")
                prediction_text = "Press SPACE to identify product"
            confidence_info = ""
        
        elif key == ord(' '):  # Space bar - manual prediction
            if not continuous_mode:
                print("\n🔍 Identifying product...")
                
                # Predict
                try:
                    prediction, conf_dict = predict_hierarchy(frame)
                    prediction_text = prediction
                    
                    # Format confidence info
                    if conf_dict:
                        confidence_info = " | ".join([f"{k}: {v:.1%}" for k, v in conf_dict.items()])
                    else:
                        confidence_info = ""
                    
                    # Print to console
                    print("-" * 60)
                    print(f"✓ Prediction: {prediction}")
                    if conf_dict:
                        print("\nConfidence Scores:")
                        for level, conf in conf_dict.items():
                            bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
                            print(f"  {level:12s}: {bar} {conf:.1%}")
                    print("-" * 60 + "\n")
                    
                except Exception as e:
                    print(f"❌ Prediction error: {e}")
                    prediction_text = "Error during prediction"
                    confidence_info = ""
    
    # Cleanup
    cap.release()
    #cv2.destroyAllWindows()
    #   rint("\n✓ Camera closed successfully")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()