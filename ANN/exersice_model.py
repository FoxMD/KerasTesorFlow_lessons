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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# create a model without over fitting it
model = Sequential()
print(X_train.shape)

# now we make it without over fitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

# size neural network on the data ... we have 78 features - so at first we will over fit
model.add(Dense(78, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(39, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(19, activation='relu'))
model.add(Dropout(rate=0.2))
# because its a binary classifications use sigmoid
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600, batch_size=256, callbacks=[early_stop])

# if validation loss gets up and training loss downs we are over fitting
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
predictions = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test, predictions))
print(explained_variance_score(y_test, predictions))

model.save('my_loan_model_sec.h5')
