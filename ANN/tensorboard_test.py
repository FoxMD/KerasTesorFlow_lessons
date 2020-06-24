import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# now we make it without over fitting
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

log_directory = "logs\\fit"
board = TensorBoard(log_directory, histogram_freq=1, write_graph=True, write_images=True, update_freq="epoch",
                    profile_batch=2)

# create a model without over fitting it
model = Sequential()
print(X_train.shape)
# size neural network on the data ... we have 30 features - so at first we will over fit
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
# because its a binary classifications use sigmoid
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600, callbacks=[early_stop, board])

# if validation loss gets up and training loss downs we are over fitting
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

predictions = model.predict(X_test)

model.save('my_cancer_model_tensorboard.h5')