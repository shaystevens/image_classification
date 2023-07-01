import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

class_names =['Plane', 'Car', 'Bird', 'Car', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model("image_classifier.model")

img = cv.imread("IMAGE.JPEG") # Add classification image here
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
resized_img = cv.resize(img, (32, 32)) 

plt.imshow(resized_img, cmap=plt.cm.binary)
plt.show()

prediction = model.predict(np.array([resized_img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')