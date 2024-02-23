import numpy as np
import tensorflow as tf
inputs = np.array([4,5,9,8,15,32], dtype=float)
outputs = np.array([25.12,31.4,56.52,50.24, 94.2, 200.96], dtype=float)
layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer_0])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(inputs, outputs, epochs=400, verbose=0)
print(model.predict([39]))
# [[243.605]]
