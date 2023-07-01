# Import the necessary modules
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load the CIFAR-10 dataset. It returns two tuples of Numpy arrays. 
# The first tuple represents the training dataset and the second one the test dataset.
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize the pixel values of the images from [0, 255] to [0, 1] for better performance 
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Define the names for the classes
class_names =['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Plot first 16 images from the training set
for i in range(16):
    plt.subplot(4, 4, i + 1)  # We'll have 4 rows and 4 columns of images
    plt.xticks([])  # Remove x-axis markers
    plt.yticks([])  # Remove y-axis markers
    plt.imshow(training_images[i], cmap=plt.cm.binary)  # Display the image in black and white
    plt.xlabel(class_names[training_labels[i][0]])  # Label the image with its class name

plt.show()  # Display the plot

# Trim the training and testing datasets to reduce computation time
training_images = training_images[:40000]
training_labels = training_labels[:40000]
testing_images = testing_images[:8000]
testing_labels = testing_labels[:8000]

# Build the model
model = models.Sequential()  # Sequential model is a linear stack of layers
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))  # First convolutional layer
model.add(layers.MaxPooling2D((2,2)))  # First pooling layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))  # Second convolutional layer
model.add(layers.MaxPooling2D((2,2)))  # Second pooling layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))  # Third convolutional layer
model.add(layers.Flatten())  # Flatten layer to convert the 3D data to 1D
model.add(layers.Dense(64, activation='relu'))  # Dense layer for performing the classification
model.add(layers.Dense(10, activation='softmax'))  # Output layer. The softmax function is used for multi-class classification

# Compile the model
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# Evaluate the model performance
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f":Loss: {loss}")  # Print the loss value
print(f"Accuracy: {accuracy}")  # Print the accuracy value

# Save the model for later use
model.save("image_classifier.model")

