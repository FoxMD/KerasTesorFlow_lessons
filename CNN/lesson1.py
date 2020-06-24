import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_cat_test = to_categorical(y_test, num_classes=10)
y_cat_train = to_categorical(y_train, num_classes=10)
x_train = x_train / 255
x_test = x_test / 255

scaled_image = x_train[0]

# batch_size, width, height, color_channels
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

model = Sequential()
# padding 28/4 = 7.0 padding
# convolution layer
model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same",
                 input_shape=(28, 28, 1), activation="relu"))
# pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))
# etc
# now flatten the image in our case: 28 * 28 = 784
model.add(Flatten())
model.add(Dense(128, activation="relu"))
# model layer should be equal to number of our classes
# output layer softmax - multi class
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=1)
model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_stop])

model.save('my_cnn_model_l1.h5')
