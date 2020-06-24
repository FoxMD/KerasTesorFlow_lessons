import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_cat_test = to_categorical(y_test, num_classes=10)
y_cat_train = to_categorical(y_train, num_classes=10)
x_train = x_train / 255
x_test = x_test / 255

scaled_image = x_train[0]

# batch_size, width, height, color_channels
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

model = keras.models.load_model('my_cnn_model_l1.h5')
model.evaluate(x_test, y_cat_test)
prediction = model.predict_classes(x_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test, prediction), annot=True)
plt.show()

my_number = x_test[0]
plt.imshow(my_number.reshape(28, 28))
plt.show()

print(model.predict_classes(my_number.reshape(1, 28, 28, 1)))


