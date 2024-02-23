import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Load the MNIST dataset and normalize pixel values
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a machine learning model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Make predictions with the untrained model
predictions = model(x_train[:1]).numpy()

# Convert logits to probabilities using softmax
probabilities = tf.nn.softmax(predictions).numpy()

# Define the loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Calculate the initial loss
initial_loss = loss_fn(y_train[:1], predictions).numpy()

# Compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test,  y_test, verbose=2)

# Wrap the model to return probabilities
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# Make predictions with probabilities
probability_predictions = probability_model(x_test[:5])

print("Initial Loss:", initial_loss)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Probabilities for the first 5 test samples:")
print(probability_predictions)
