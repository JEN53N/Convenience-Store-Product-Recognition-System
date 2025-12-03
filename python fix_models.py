"""
Script to fix legacy Keras models with batch_shape compatibility issues
Run this once to convert all models to a compatible format
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import h5py
import shutil
from datetime import datetime

MODEL_DIR = "models"
BACKUP_DIR = "models_backup"

# Model files to fix
MODEL_FILES = [
    "model_level1_mobilenet.h5",
    "model_level2_foodnbeverages.h5",
    "model_level2_personalcare.h5",
    "model_level3_4_chips.h5",
    "model_level3_4_biscuits.h5",
    "model_level3_4_drinks.h5",
    "model_level3_4_oralcare.h5",
    "model_level3_4_bodycare.h5"
]

def backup_models():
    """Create backup of original models"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, timestamp)
    os.makedirs(backup_path)
    
    print(f"\n📦 Creating backup in: {backup_path}")
    for model_file in MODEL_FILES:
        src = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(src):
            dst = os.path.join(backup_path, model_file)
            shutil.copy2(src, dst)
            print(f"  ✓ Backed up {model_file}")
    
    return backup_path

def fix_model(model_path):
    """Fix a single model file"""
    print(f"\n🔧 Fixing: {os.path.basename(model_path)}")
    
    temp_path = model_path + ".tmp"
    
    try:
        # Method 1: Load and re-save with compile=False
        print("  Attempting direct load...")
        try:
            model = keras.models.load_model(model_path, compile=False)
            model.save(temp_path, save_format='h5')
            os.replace(temp_path, model_path)
            print("  ✓ Fixed successfully (Method 1)")
            return True
        except Exception as e:
            print(f"  ⚠ Method 1 failed: {str(e)[:50]}")
        
        # Method 2: Load with custom objects
        print("  Attempting load with custom objects...")
        try:
            class CustomInputLayer(keras.layers.InputLayer):
                def __init__(self, *args, **kwargs):
                    if 'batch_shape' in kwargs:
                        batch_shape = kwargs.pop('batch_shape')
                        if batch_shape and len(batch_shape) > 1:
                            kwargs['input_shape'] = batch_shape[1:]
                    super().__init__(*args, **kwargs)
            
            custom_objects = {'InputLayer': CustomInputLayer}
            model = keras.models.load_model(
                model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            model.save(temp_path, save_format='h5')
            os.replace(temp_path, model_path)
            print("  ✓ Fixed successfully (Method 2)")
            return True
        except Exception as e:
            print(f"  ⚠ Method 2 failed: {str(e)[:50]}")
        
        # Method 3: Manual h5 file manipulation
        print("  Attempting manual h5 manipulation...")
        try:
            import json
            
            with h5py.File(model_path, 'r+') as f:
                # Check if model_config exists
                if 'model_config' in f.attrs:
                    config_str = f.attrs['model_config']
                    if isinstance(config_str, bytes):
                        config_str = config_str.decode('utf-8')
                    
                    config = json.loads(config_str)
                    
                    # Fix batch_shape in layers
                    modified = False
                    if 'config' in config and 'layers' in config['config']:
                        for layer in config['config']['layers']:
                            if 'config' in layer and 'batch_shape' in layer['config']:
                                batch_shape = layer['config']['batch_shape']
                                if batch_shape and len(batch_shape) > 1:
                                    layer['config']['input_shape'] = batch_shape[1:]
                                del layer['config']['batch_shape']
                                modified = True
                    
                    if modified:
                        # Write back modified config
                        f.attrs['model_config'] = json.dumps(config)
                        print("  ✓ Fixed successfully (Method 3)")
                        return True
        except Exception as e:
            print(f"  ⚠ Method 3 failed: {str(e)[:50]}")
        
        print("  ❌ All methods failed")
        return False
        
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def verify_model(model_path):
    """Verify that model can be loaded"""
    try:
        model = keras.models.load_model(model_path, compile=False)
        # Try a dummy prediction
        import numpy as np
        dummy_input = np.random.random((1, 224, 224, 3)).astype('float32')
        _ = model.predict(dummy_input, verbose=0)
        return True
    except Exception as e:
        print(f"  ⚠ Verification failed: {str(e)[:50]}")
        return False

def main():
    print("="*70)
    print("  MODEL COMPATIBILITY FIX SCRIPT")
    print("="*70)
    print("\nThis script will:")
    print("1. Backup all existing models")
    print("2. Fix batch_shape compatibility issues")
    print("3. Verify fixed models can load properly")
    print("\n" + "="*70)
    
    input("\nPress ENTER to continue or CTRL+C to cancel...")
    
    # Backup original models
    backup_path = backup_models()
    
    # Fix each model
    print("\n" + "="*70)
    print("FIXING MODELS")
    print("="*70)
    
    fixed_count = 0
    failed_models = []
    
    for model_file in MODEL_FILES:
        model_path = os.path.join(MODEL_DIR, model_file)
        
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_file} (not found)")
            continue
        
        if fix_model(model_path):
            # Verify the fix
            print(f"  Verifying {model_file}...", end=" ")
            if verify_model(model_path):
                print("✓")
                fixed_count += 1
            else:
                print("❌")
                failed_models.append(model_file)
        else:
            failed_models.append(model_file)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Successfully fixed: {fixed_count}/{len(MODEL_FILES)}")
    print(f"✓ Backup location: {backup_path}")
    
    if failed_models:
        print(f"\n❌ Failed to fix {len(failed_models)} model(s):")
        for model in failed_models:
            print(f"  - {model}")
        print("\nThese models may need to be retrained or converted manually.")
    else:
        print("\n🎉 All models fixed successfully!")
        print("\nYou can now run: python camera_predict_hierarchy.py")
    
    print("="*70)

if __name__ == "__main__":
    main()