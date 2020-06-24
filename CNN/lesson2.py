import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# pre processing
x_train = x_train / 255
x_test = x_test / 255

# y is continuous we need it to be categorical
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

model = Sequential()
# convolution layer, 32,32,3 = 3072
model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same",
                 input_shape=(32, 32, 3), activation="relu"))
# pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))
# convolution layer2
model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same",
                 input_shape=(32, 32, 3), activation="relu"))
# pooling layer 2
model.add(MaxPool2D(pool_size=(2, 2)))

# now flatten the image in our case: 28 * 28 = 784
model.add(Flatten())
# added complexity calls for more neurons
model.add(Dense(256, activation="relu"))
# model layer should be equal to number of our classes
# output layer softmax - multi class
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=2)
model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_stop])

# if validation loss gets up and training loss downs we are over fitting
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

model.save('my_cnn_model_cfr10.h5')
