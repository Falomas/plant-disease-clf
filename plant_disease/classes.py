# Восемь классов. Имена папок как в raw/color репозитория PlantVillage.

SOURCE_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
SOURCE_VARIANT = "raw/color"

CLASS_FOLDERS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

# Доли и зерно зафиксированы заданием.
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
