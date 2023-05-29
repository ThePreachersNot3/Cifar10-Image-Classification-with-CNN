#importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

#got a dataset from keras
(X_train,y_train),(X_test,y_test) = datasets.cifar10.load_data()
#reshaped our y_train so as to be able to make use of it as an index
#if we used it directly we would be making use of a list which wont work
y_train = y_train.reshape(-1,)
#got the details of each category of the images on 
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#data visualization
#a function that made use of X_train and the index
#y_train and the index, then passed into the classes list as an index
def plot_samp(X,y, index):
    plt.figure(figsize=(10,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()
plot_samp(X_train, y_train,0)

#data normalisation

X_train = X_train/255
X_test = X_test/255

#CNN model
model = keras.Sequential([
    #kernel size here is used to specify the height and width of the 2D convolution window
    #filter sets the number of convolution filters in the layer
    keras.layers.Conv2D(filters=32, activation='relu', kernel_size=(3,3), input_shape=(32,32,3)),
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3)),
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    
)

model.fit(X_train, y_train, epochs=5)