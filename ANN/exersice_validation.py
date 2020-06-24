import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../DATA/lending_club_info.csv", index_col="LoanStatNew")


def feat_info(col_name):
    print(data.loc[col_name]["Description"])


df = pd.read_csv("../DATA/myLC.csv")

from sklearn.model_selection import train_test_split

df = df.drop("loan_status", axis=1)
X = df.drop("loan_repaid", axis=1).values
y = df["loan_repaid"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow import keras
model = keras.models.load_model('my_loan_model_sec.h5')

from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))

import random
random.seed(101)
# df = df.drop("Unnamed: 0", axis=1)
random_id = random.randint(0, len(df))

new_customer = df.drop("loan_repaid", axis=1).iloc[random_id]
print(new_customer)

new_customer = scaler.transform(new_customer.values.reshape(1, 79))
print(model.predict_classes(new_customer))


