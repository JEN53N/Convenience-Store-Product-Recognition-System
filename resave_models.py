from tensorflow.keras.models import load_model

# List your old models here
models = [
    "models/model_level1_mobilenet.h5",
    "models/model_level2_foodnbeverages.h5",
    "models/model_level2_personalcare.h5",
    "models/model_level3_4_chips.h5",
    "models/model_level3_4_biscuits.h5",
    "models/model_level3_4_drinks.h5",
    "models/model_level3_4_oralcare.h5",
    "models/model_level3_4_bodycare.h5"
]

for m in models:
    print(f"Resaving {m} ...")
    model = load_model(m, compile=False)  # load old model
    model.save(m)  # overwrite with new format
    print(f"{m} resaved successfully.")
