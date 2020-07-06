import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix
data_dir = "../cell_images"

from matplotlib.image import imread
test_path = data_dir+"\\test\\"

from tensorflow.keras.preprocessing import image
mypath = os.listdir(test_path+"parasitized")[0]
para_cell = test_path+"parasitized\\"+"C100P61ThinF_IMG_20150918_144348_cell_144.png"
plt.imshow(imread(para_cell))
plt.show()
