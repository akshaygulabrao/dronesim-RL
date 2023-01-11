import numpy as np
import tensorflow as tf


inputs = tf.keras.layers.Input(shape=(1,))
x = tf.keras.layers.Dense(100, activation='sigmoid')(inputs)
x = tf.keras.layers.Dense(100, activation='sigmoid')(x)
outputs = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])


x = np.random.randint(1, 400, (1000, 1))
x = np.random.random((1000,1))
# y = np.random.randint(0, 2, (2, 2))
y = np.cos(x*np.pi)

model.fit(x,y,epochs=500)