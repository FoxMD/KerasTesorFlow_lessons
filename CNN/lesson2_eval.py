import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# pre processing
x_train = x_train / 255
x_test = x_test / 255

# y is continuous we need it to be categorical
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

model = keras.models.load_model('my_cnn_model_cfr10.h5')
model.evaluate(x_test, y_cat_test)
prediction = model.predict_classes(x_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test, prediction), annot=True)
plt.show()

my_picture = x_test[60]
plt.imshow(my_picture.reshape(32, 32, 3))
plt.show()

print(model.predict_classes(my_picture.reshape(1, 32, 32, 3)))
print(y_test[60])

