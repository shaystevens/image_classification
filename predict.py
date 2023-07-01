# Import necessary libraries
import cv2 as cv # OpenCV for image processing
import numpy as np # NumPy for numerical computations
import matplotlib.pyplot as plt # Matplotlib for plotting
from tensorflow.keras import datasets, layers, models # TensorFlow and Keras for machine learning

# Define the class names. These are the categories that the model has been trained to recognize.
class_names =['Plane', 'Car', 'Bird', 'Car', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load the previously trained model from a file
model = models.load_model("image_classifier.model")

# Load the image to be classified
img = cv.imread("IMAGE.JPEG")

# Convert the image from BGR color format (which OpenCV uses) to RGB format
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Resize the image to the size that the model expects (32x32 pixels in this case)
resized_img = cv.resize(img, (32, 32)) 

# Display the resized image using Matplotlib
plt.imshow(resized_img, cmap=plt.cm.binary)
plt.show()

# Use the model to predict the class of the image. The image is first converted to an array,
# and the pixel intensities are normalized to be between 0 and 1 by dividing by 255.
prediction = model.predict(np.array([resized_img]) / 255)

# The model's prediction is an array of probabilities for each class. We take the index of 
# the highest probability using np.argmax, which gives us the predicted class.
index = np.argmax(prediction)

# Finally, we print the name of the predicted class
print(f'Prediction is {class_names[index]}')
