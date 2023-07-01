# Image Classifier

This project is a simple image classifier trained on the CIFAR-10 dataset using a convolutional neural network (CNN). The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes include plane, car, bird, cat, deer, dog, frog, horse, ship, and truck. 

The project is divided into two main parts:

1. Training the model
2. Using the model for prediction

## Requirements
- Python 3
- OpenCV
- TensorFlow 2.x
- Matplotlib
- NumPy

## Part 1: Training the Model
Firstly, the CIFAR-10 dataset is loaded and normalized to have pixel values between 0 and 1. A CNN model is created using TensorFlow and Keras and trained on a subset of the CIFAR-10 dataset. The model architecture includes three convolutional layers with max-pooling followed by a dense layer and an output layer.

The model is then compiled with the Adam optimizer and the sparse categorical cross entropy loss function. It is then trained for 10 epochs.

After training, the model's performance is evaluated on the testing dataset. The loss and accuracy values are printed and the model is saved for later use.

## Part 2: Using the Model for Prediction
A saved model is loaded and used to predict the class of a new image. The new image is loaded using OpenCV, converted to RGB color format, and resized to 32x32 pixels to match the input size that the model expects.

The image is displayed and then passed through the model to get a prediction. The class with the highest prediction probability is taken as the predicted class and printed.

## How to Run
To run the project:

1. Train the model using the training script.
```
python train.py
```

2. Use the trained model to classify a new image with the prediction script. Note: replace "IMAGE.JPEG" with the name of the image you want to classify.
```
python predict.py
```

## Future Work
The performance of the model could potentially be improved by using more training data, data augmentation, a more complex model architecture, training for more epochs, tuning the learning rate, adding batch normalization layers, or using transfer learning. These are areas for future exploration.
