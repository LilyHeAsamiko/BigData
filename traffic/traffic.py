import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(tf.convert_to_tensor(x_train, dtype = tf.float32), tf.convert_to_tensor(y_train, dtype = tf.float32), epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(tf.convert_to_tensor(x_test, dtype = tf.float32),  tf.convert_to_tensor(y_test, dtype = tf.float32), verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    try:
        images = []
        labels = []
        #p = data_dir.split(os.sep.split(data_dir)[0]) 
        h,t=os.path.split(data_dir)#os.path.realpath("E:\BigData\traffic\traffic.py")
        files = os.listdir(r"E:\BigData\traffic\gtsrb-small")
        #path_to_folder= 欧式。path join（‘.’,f"{data_dir}"）
        for img_d in files:
            lb_d = os.path.join('E:','BigData','traffic','gtsrb-small',img_d) #os.sep.join(["E:","BigData","traffic","gtsrb-small",'0'])
            #print(lb_d)
            lb_p = os.listdir(lb_d)
            #print(lb_p)
            for img_p in lb_p:
                path = os.path.join('E:','BigData','traffic','gtsrb-small',img_d,img_p)
                #img = cv2.imread(path)[0:IMG_WIDTH,0:IMG_HEIGHT,0:3]
                img = cv2.imread(path)
                img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
                images.append(img)
                labels.append(int(img_d))      
        print(len(images),len(labels))
        return (images, labels)
    except:
        raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    try:
        model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT,3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')#sigmoid
        ])  
        model.compile(optimizer='adam',
                  loss='categorical_crossentropy',#'catogory_crossentropy',
                  metrics=['accuracy'])
        return model
    except:
        raise NotImplementedError


if __name__ == "__main__":
    main()
