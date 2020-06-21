import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../DATA/cancer_classification.csv")

# train split
X = df.drop("benign_0__mal_1", axis=1).values
y = df["benign_0__mal_1"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow import keras
model_d = keras.models.load_model('my_cancer_model_dropped.h5')
model_i = keras.models.load_model('my_cancer_model_optimized.h5')
model_o = keras.models.load_model('my_cancer_model_overfitted.h5')

from sklearn.metrics import classification_report, confusion_matrix
predictions = model_d.predict_classes(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
