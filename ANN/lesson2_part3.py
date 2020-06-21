import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../DATA/kc_house_data.csv")
df = df.drop("id", axis=1)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].apply(lambda date: date.year)
df["month"] = df["date"].apply(lambda date: date.month)
df = df.drop("date", axis=1)
df = df.drop("zipcode", axis=1)

X = df.drop("price", axis=1).values
y = df["price"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow import keras
model = keras.models.load_model('my_house_price_model.h5')

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test, predictions))
print(explained_variance_score(y_test, predictions))

single_house = df.drop("price", axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))
model.predict(single_house)

print(df.head(1))
