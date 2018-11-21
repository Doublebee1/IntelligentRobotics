## TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

## Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
## The train_images and train_labels arrays are the training set—the data the model uses to learn.
## The model is tested against the test set, the test_images, and test_labels arrays.

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images.shape
## How many data in train set
#print("train_labels:" ,len(train_labels))

test_images.shape
## How many data in test set
#print("test_labels:" , len(test_labels))

## Draw picture with matplotlib
plt.figure()
plt.imshow(train_images[5000])
## You can change array number to check other data.
plt.colorbar()
plt.grid(False)
## plt.show() is positioned next all of plt instruction.
#plt.show()

## We scale these values to a range of 0 to 1 before feeding to the neural network model. 
## For this, cast the datatype of the image components from an integer to a float, and divide by 255. 
## Here's the function to preprocess the images.
## It's important that the training set and the testing set are preprocessed in the same way.
train_images = train_images / 255.0
test_images = test_images / 255.0


##--PREPROCESS THE DATA
## Display the first 25 images from the training set and display the class name below each image.
## Verify that the data is in the correct format and we're ready to build and train the network.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

##--BUILD THE MODEL
## The basic building block of a neural network is the layer. 
## Layers extract representations from the data fed into them. And, hopefully, these representations are more meaningful for the problem at hand.
## Most of deep learning consists of chaining together simple layers. Most layers, like tf.keras.layers.Dense, have parameters that are learned during training.
model = keras.Sequential([
## Flatten() transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
keras.layers.Flatten(input_shape=(28, 28)),
## Dense() These are densely-connected, or fully-connected, neural layers
keras.layers.Dense(128, activation=tf.nn.relu),
keras.layers.Dense(10, activation=tf.nn.softmax)
])

##--COMPILE THE MODEL
## Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
## Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
## Optimizer —This is how the model is updated based on the data it sees and its loss function.
## Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

##--TRAIN THE MODEL              
## To start training, call the model.fit() method—the model is "fit" to the training data:
model.fit(train_images, train_labels, epochs=5)

##--EVALUATE ACCURACY
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

##--MAKE PREDICTIONS
## With the model trained, we can use it to make predictions about some images.
predictions = model.predict(test_images)
## Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
predictions[0]

## A prediction is an array of 10 numbers.
## These describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing.
## We can see which label has the highest confidence value:
print(np.argmax(predictions[0]), test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


## Let's look at the 0th image, predictions, and prediction array.
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
#plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
#plt.show()

## Plot the first X test images, their predicted label, and the true label
## Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

## Grab an image from the test dataset
img = test_images[0]
print(img.shape)

## Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])