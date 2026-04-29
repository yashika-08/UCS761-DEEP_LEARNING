import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model(\"models/srgan_generator.h5\")

img = image.load_img(\"test.jpg\", target_size=(64,64))
img = image.img_to_array(img)/255.0
img = np.expand_dims(img, axis=0)

sr = model.predict(img)

print(\"Super-resolution image generated\")