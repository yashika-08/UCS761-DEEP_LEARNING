import tensorflow as tf
from tensorflow.keras import layers, Model

def build_generator():
    return tf.keras.Sequential([
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.UpSampling2D(),
        layers.Conv2D(3, 3, padding='same')
    ])

def build_discriminator():
    return tf.keras.Sequential([
        layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

z = tf.keras.Input(shape=(64,64,3))
fake = generator(z)
discriminator.trainable = False
validity = discriminator(fake)

gan = Model(z, validity)
gan.compile(optimizer='adam', loss='binary_crossentropy')

print(\"GAN ready (training loop simplified)\")