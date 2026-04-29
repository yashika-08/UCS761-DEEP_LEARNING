import tensorflow as tf
from tensorflow.keras import layers

def residual_block(x):
    y = layers.Conv2D(64,3,padding='same')(x)
    y = layers.ReLU()(y)
    y = layers.Conv2D(64,3,padding='same')(y)
    return layers.add([x,y])

def build_srgan():
    inputs = tf.keras.Input(shape=(64,64,3))
    x = layers.Conv2D(64,9,padding='same')(inputs)
    for _ in range(5):
        x = residual_block(x)
    x = layers.UpSampling2D()(x)
    outputs = layers.Conv2D(3,9,padding='same')(x)
    return tf.keras.Model(inputs, outputs)

model = build_srgan()
model.compile(optimizer='adam', loss='mse')

print(\"SRGAN model ready\")