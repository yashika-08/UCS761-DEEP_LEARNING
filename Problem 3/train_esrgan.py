import tensorflow as tf
from tensorflow.keras import layers

def dense_block(x):
    for _ in range(3):
        y = layers.Conv2D(64,3,padding='same',activation='relu')(x)
        x = layers.Concatenate()([x,y])
    return x

def build_esrgan():
    inputs = tf.keras.Input(shape=(64,64,3))
    x = layers.Conv2D(64,3,padding='same')(inputs)
    for _ in range(3):
        x = dense_block(x)
    x = layers.UpSampling2D()(x)
    outputs = layers.Conv2D(3,3,padding='same')(x)
    return tf.keras.Model(inputs, outputs)

model = build_esrgan()
model.compile(optimizer='adam', loss='mse')

print(\"ESRGAN model ready\")