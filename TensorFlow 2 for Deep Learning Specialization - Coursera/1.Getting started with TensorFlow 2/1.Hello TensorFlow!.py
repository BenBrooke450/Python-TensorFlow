





# Import TensorFlow
import tensorflow as tf
import kagglehub


# Check its version

print(tf.__version__)
#2.17.0





#Download latest version
#path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
#print("Path to dataset files:", path)


#'/Users/benjaminbrooke/.cache/kagglehub/datasets/oddrationale/mnist-in-csv/versions/2/mnist_test.csv'


# Train a feedforward neural network for image classification

import numpy as np

print('Loading data...\n')
data = np.genfromtxt('/Users/benjaminbrooke/.cache/kagglehub/datasets/oddrationale/mnist-in-csv/versions/2/mnist_test.csv', delimiter=',')
print('MNIST dataset loaded.\n')

x_train = data[:, 1:]
y_train = data[:, 0]
x_train = x_train/255.

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Training model...\n')
model.fit(x_train, y_train, epochs=3, batch_size=32)

print('Model trained successfully!')