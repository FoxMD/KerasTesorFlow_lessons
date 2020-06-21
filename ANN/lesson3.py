import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../DATA/cancer_classification.csv")
# classification task - check for NaN or Nulls
# print(df.info())
# print(df.describe().transpose())

# exploratory data analyses with graphics
# sns.countplot(x="benign_0__mal_1", data=df)
# plt.show()
# df.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")
# sns.heatmap(df.corr())
# plt.show()

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# create a model without over fitting it
model = Sequential()
print(X_train.shape)
# size neural network on the data ... we have 30 features - so at first we will over fit
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
# because its a binary classifications use sigmoid
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600)

# if validation loss gets up and training loss downs we are over fitting
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test, predictions))
print(explained_variance_score(y_test, predictions))

model.save('my_cancer_model_overfitted.h5')

# now we make it without over fitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

# create a model without over fitting it
model = Sequential()
print(X_train.shape)
# size neural network on the data ... we have 30 features - so at first we will over fit
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
# because its a binary classifications use sigmoid
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600, callbacks=[early_stop])

# if validation loss gets up and training loss downs we are over fitting
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test, predictions))
print(explained_variance_score(y_test, predictions))

model.save('my_cancer_model_optimized.h5')

# now we make it with dropout
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

# create a model without over fitting it
model = Sequential()
print(X_train.shape)
# size neural network on the data ... we have 30 features - so at first we will over fit
model.add(Dense(30, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(rate=0.5))
# because its a binary classifications use sigmoid
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600, callbacks=[early_stop])

# if validation loss gets up and training loss downs we are over fitting
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test, predictions))
print(explained_variance_score(y_test, predictions))

model.save('my_cancer_model_dropped.h5')