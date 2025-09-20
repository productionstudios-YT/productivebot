import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(10,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

X = np.random.rand(100, 10)
y = np.random.randint(0, 2, (100,))

model.fit(X, y, epochs=5, verbose=0)
model.save("chat_model.h5")
print("Dummy model saved as chat_model.h5")
