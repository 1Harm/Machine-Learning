import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
tfds.list_builders()
Dataset = tfds.builder('rock_paper_scissors')
info = Dataset.info

print(info)
# Loading the dataset
ds_train = tfds.load(name="rock_paper_scissors", split="train")
ds_test = tfds.load(name="rock_paper_scissors", split="test")

# Iterating over the images and storing
# it in train and test datas
train_images = np.array([image['image'].numpy()[:, :, 0]
                         for image in ds_train])
train_labels = np.array([image['label']
                        .numpy() for image in ds_train])

test_images = np.array([image['image'].numpy()[:, :, 0] for image in ds_test])
test_labels = np.array([image['label'].numpy() for image in ds_test])

# Reshaping the images
train_images = train_images.reshape(2520, 300, 300, 1)
test_images = test_images.reshape(372, 300, 300, 1)

# Changing the datatype
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# getting the values down to 0 and 1
train_images /= 255
test_images /= 255

# A convolutional neural network

# Defining the model
model = keras.Sequential([
    keras.layers.Conv2D(64, 3, activation='relu',
                        input_shape=(300, 300, 1)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Fitting the model with data
model.fit(train_images, train_labels, epochs=5,
          batch_size=32)
model.evaluate(test_images, test_labels)

# A better convolutional neural network

# Model defining
model = keras.Sequential([
    keras.layers.AveragePooling2D(6, 3,
                                  input_shape=(300, 300, 1)),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compiling a model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Fitting the model
model.fit(train_images, train_labels, epochs=5,
          batch_size=32)