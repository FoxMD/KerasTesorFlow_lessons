import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../DATA/fake_reg.csv')
print(df.head())

# take y and x data from dataframe
X = df[['feature1', 'feature2']].values
y = df['price'].values

# split data between test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(X_train.shape)
# print(X_test.shape)

# scale data between 0 and 1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create model using tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# two methods
# model = Sequential([Dense(4, activation='relu'), Dense(2, activation='relu'), Dense(1)])
# better method
model = Sequential()

model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')
model.fit(x=X_train, y=y_train, epochs=600)

# save the model
from tensorflow.keras.models import load_model
model.save('my_gem_model_new.h5')

# prepare model for eval
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

model.evaluate(x=X_test, y=y_test, verbose=0)

test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=['Test true Y'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['Test True Y', 'Model Predictions']
sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions']))

plt.show()
