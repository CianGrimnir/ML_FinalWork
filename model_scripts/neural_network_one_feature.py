#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

df = pd.read_csv("dublinbikes_20200101_20200401.csv")
df1 = df.copy()
df1['STATUS'] = df1['STATUS'].astype('category')
df1['LAST UPDATED'] = pd.to_datetime(df1['LAST UPDATED'])
df1['TIME'] = pd.to_datetime(df1['TIME'])
df1['formatted_time'] = df1['TIME'].dt.floor('h')
df1['day_of_week'] = df1['formatted_time'].dt.strftime('%A')
df1['day_of_month'] = df1['formatted_time'].dt.strftime('%d').astype(np.int64)
df1['hour'] = df1['formatted_time'].dt.strftime('%H').astype(np.int64)
df1['month'] = df1['formatted_time'].dt.strftime('%m').astype(np.int64)
df1['week'] = df1['formatted_time'].dt.strftime('%w').astype(np.int64)
numeric_columns = df1.select_dtypes(['int64']).columns
print(df1[numeric_columns].describe().T)
print(df1.select_dtypes(['category']).describe().T)

ts = pd.to_datetime('2020/02/01')
te = pd.to_datetime('2020/04/01')
mask = (df1['TIME'] >= ts) & (df1['TIME'] <= te)
pd.options.mode.chained_assignment = None
suburb_point = "Merrion Square South"
# uncomment below line for running for Grangegorman Lower South
# suburb_point = "Grangegorman Lower (South)"
suburb_df = df1.loc[mask]
suburb_df = suburb_df[suburb_df['ADDRESS'] == suburb_point]
suburb_dataset = suburb_df[['TIME', 'AVAILABLE BIKES']]
suburb_dataset['date'] = suburb_dataset['TIME'].dt.floor('T')
suburb_dataset = suburb_dataset.reset_index()
suburb_dataset.drop('TIME', axis=1, inplace=True)
time = suburb_dataset['date']
bikes = suburb_dataset['AVAILABLE BIKES']
# plot 2 week data for Merrion Square South region
fig = plt.figure()
ax = fig.add_subplot(111)
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.scatter(time, bikes, color="red", marker=".", label='bike count')
ax.set_xlabel("date")
ax.set_ylabel("bike available")
fig.autofmt_xdate()
ax.grid(True)
ax.set_title(f'Bike usage for {suburb_point}')
plt.show()

t_full = pd.array(pd.DatetimeIndex(suburb_df.iloc[:, 1]).astype(np.int64)) / 1000000000
dt = t_full[1] - t_full[0]
qrange = [2, 6, 12]
lag = 3
stride = 1
plt.close("all")
fig, ax = plt.subplots(3, 2)
ax_index = 0
q_score = {}
for q in qrange:
    yy, tt, XX = utils.get_feature_data(bikes, time, lag, q, stride, dt)
    train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)

    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=9))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    history = model.fit(XX[train], yy[train], validation_data=(XX[test], yy[test]), epochs=5000, batch_size=100, verbose=0,
                        callbacks=[es])
    ypred = model.predict(XX)

    ax[ax_index, 0].set_title(f'loss of the neural model for {q}')
    ax[ax_index, 0].plot(history.history['loss'], label='train')
    ax[ax_index, 0].plot(history.history['val_loss'], label='test')
    ax[ax_index, 0].set_xlabel("epoch")
    ax[ax_index, 0].set_ylabel("loss")
    ax[ax_index, 0].legend()
    ax[ax_index, 0].axis('tight')

    ax[ax_index, 1].set_title(f'Training Set for {q}')
    ax[ax_index, 1].scatter(time, bikes, color="deepskyblue", marker='.')
    ax[ax_index, 1].scatter(tt, ypred, color="lightgreen", marker='.')
    ax[ax_index, 1].set_xlabel("time (days)")
    ax[ax_index, 1].set_ylabel("#bikes")
    ax[ax_index, 1].legend(["training data", "predictions"], loc="upper right")
    scores = r2_score(yy, ypred)
    print(q, scores)
    q_score[q] = scores
    ax_index += 1

for q, score in q_score.items():
    print(f"q value {q} - score {score}")

fig.autofmt_xdate()
plt.show()
