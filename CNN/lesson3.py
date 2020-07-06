import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.python.layers.core import Dense

data_dir = "../cell_images"

from matplotlib.image import imread
test_path = data_dir+"\\test\\"
train_path = data_dir+"\\train\\"

# each image is not the same size
dim1 = []
dim2 = []

para_cell_test = test_path+"parasitized\\"
uninfected_cell_test = test_path+"uninfected\\"

para_cell_train = train_path+"parasitized\\"
uninfected_cell_train = train_path+"uninfected\\"

# for image_filename in os.listdir(uninfected_cell_test):
#    img = imread(uninfected_cell_test+image_filename)
#    d1, d2, colors = img.shape
#    dim1.append(d1)
#    dim2.append(d2)

# we need to resize the images to mean 130,92/130,75 -> arround 50k
image_shape = (130, 130, 3)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, rescale=None,
                               shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode="nearest")
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=image_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=image_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=2)
batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path, target_size=image_shape[:2], color_mode="rgb",
                                                batch_size=batch_size, class_mode="binary")
test_image_gen = image_gen.flow_from_directory(test_path, target_size=image_shape[:2], color_mode="rgb",
                                                batch_size=batch_size, class_mode="binary", shuffle=False)

result = model.fit_generator(train_image_gen, epochs=20, validation_data=test_image_gen, callbacks=[early_stop])

# if validation loss gets up and training loss downs we are over fitting
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

model.save('my_malaria_model.h5')
