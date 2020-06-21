import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../DATA/kc_house_data.csv")
# drop unnecessary data
df = df.drop("id", axis=1)
# make datetime from date
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].apply(lambda date: date.year)
df["month"] = df["date"].apply(lambda date: date.month)
# print(df.head())
# check if date matter for sale
# sns.boxplot(x="month", y="price", data=df)
# plt.show()
# boxplot was the same for all so try numbers for small changes
# df.groupby("month").mean()["price"].plot()
# plt.show()
# there is a change in moths so let them be there and remove date
df = df.drop("date", axis=1)
# zip code could be a problem because it would be taken as a increasing number
# df["zipcode"].value_counts()
# 70 categories are to much for grouping
df = df.drop("zipcode", axis=1)
# year renovated could be also a problem because most of the values are zeroes
# feature engineering could be said 0 was not renovated else was renovated - but we are lucky, higher year - more value
# the same for basement 0 - has no basement, else has basement... so we can keep it as continues
X = df.drop("price", axis=1).values
y = df["price"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# make scaling on the training set to prevent data leakage on the test set
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# size neural network on the data ... we have 19 features - 19 neurons would be nice
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=128, epochs=600)

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test, predictions))
print(explained_variance_score(y_test, predictions))

model.save('my_house_price_model.h5')
