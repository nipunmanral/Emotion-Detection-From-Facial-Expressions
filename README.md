# Emotion Detection From Facial Expressions

## Objective
Facial detection, landmark detection, gender classification and emotion recognition, based on visual appearances have great potential for real-world applications such as automatic cosmetics, entertainment, human-app interaction, advertisement promotion etc. In this project, we focus on building a Convolutional Neural Network model using Keras for emotion classification from image inputs.

## Environment Setup
Download the codebase and open up a terminal in the root directory. Make sure python 3.6 is installed in the current environment. Then execute

    pip install -r requirements.txt

This should install all the necessary packages for the code to run.

## Dataset
The dataset consists of an image directory with images (color, black & white) in the .jpg format of human faces. These images have not been uploaded here due to size constraints. The train.csv file maps an image in the images directory with a facial expression. We use the Pillow library for image manipulation and augmentation (resizing, converting to black and white, transposing, cropping etc.). The dataset used for this project contained 12,992 images which are split into training, validation and test dataset.

## Code Information
We use a Python generator to train the model with sparse_categorical_crossentropy as the loss function and adam as the optimizer. ReLU is used as the activation function in the hidden layers and Softmax in the output layer. Dropout and Batch normalisation have also been used in the convolutional neural network.

## Evaluation
Models are scored on classification accuracy. Since this is a multi-class problem, the score is simply the number of correct classifications divided by the total number of elements in the test set.
