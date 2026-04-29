import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset/NEU-DET/train/images")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 6

# DATA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Classes:", train_gen.class_indices)

# MODEL BUILDER
def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# TRAIN FUNCTION
def train_model(name, base_fn, filename):
    print(f"\nTraining {name}...")

    base_model = base_fn(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model = build_model(base_model)

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    path = os.path.join(MODEL_DIR, filename)
    model.save(path)
    print(f"{name} saved at {path}")

    loss, acc = model.evaluate(val_gen)
    print(f"{name} Accuracy: {acc:.4f}")

# RUN
if __name__ == "__main__":
    train_model("ResNet50", ResNet50, "resnet50_defect.h5")
    train_model("InceptionV3", InceptionV3, "inception_defect.h5")
    train_model("EfficientNetB0", EfficientNetB0, "efficientnet_defect.h5")