#!/usr/bin/env python

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

df = pd.read_csv("dublinbikes_20200101_20200401.csv")
eval_df = df.copy()
# plot top 20 stations with high Bike Usage
eval_df['TIME'] = pd.to_datetime(eval_df['TIME'])
eval_df['date'] = eval_df['TIME'].dt.floor('D')
eval_df = eval_df.groupby(['STATION ID', 'date', 'STATUS', 'ADDRESS'])\
    .agg({'AVAILABLE BIKES': 'sum', 'AVAILABLE BIKE STANDS': 'sum'})
eval_df = eval_df.reset_index(level=[0, 1, 2, 3])
top_address = eval_df.groupby('ADDRESS')['AVAILABLE BIKE STANDS'].sum().nlargest(20).index.tolist()
top_df = pd.DataFrame(columns=eval_df.columns)
for address in top_address:
    res = eval_df[eval_df['ADDRESS'] == address]
    top_df = top_df.append(res)
# normal plot show
top_df.set_index('date', inplace=True)
fig, ax = plt.subplots(2, 2)


# minimize the gap between time interval
def plot_graph(ax, top_address, label):
    for address in top_address:
        ax.plot(top_df[top_df['ADDRESS'] == address].index,
                top_df[top_df['ADDRESS'] == address]['AVAILABLE BIKE STANDS'], label=address)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.grid(True)
    ax.set_title(f'top {label} dublin station against cycle count')
    ax.set_xlabel("Day")
    ax.set_ylabel("Total Cycle Usage")
    ax.legend(loc="upper left")
    return ax


ax[0, 0] = plot_graph(ax[0, 0], top_address[0:5], '5')
ax[0, 1] = plot_graph(ax[0, 1], top_address[5:10], '5-10')
ax[1, 0] = plot_graph(ax[1, 0], top_address[10:15], '10-15')
ax[1, 1] = plot_graph(ax[1, 1], top_address[15:20], '15-20')
fig.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()

# Add additional features to the dataset
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
# extract features from 2nd Feb to 1st Apr for station - Merrion Square/Grangegorman
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

# identify the lag value
day_of_month = suburb_df['day_of_month']
hour = suburb_df['hour']
t_full = pd.array(pd.DatetimeIndex(suburb_df.iloc[:, 1]).astype(np.int64)) / 1000000000
dt = t_full[1] - t_full[0]
q = 2
lag_range = [1, 2, 3, 4]
stride = 1
Ci = 50
scores = []
errors = []
for lag in lag_range:
    yy, _, XX = utils.get_many_feature_data(bikes, time, q, lag, stride, dt, day_of_month, hour)
    train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
    a = 1 / (2 * Ci)
    print("lag value - ", lag)
    model = Lasso(alpha=a).fit(XX[train], yy[train])
    y_pred = model.predict(XX)
    score = model.score(XX[test], yy[test])
    mse = mean_squared_error(yy, y_pred)
    errors.append(mse)
    r2 = r2_score(yy, y_pred)
    scores.append(r2)
    print("SCORE: {0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}".format(score, mse, np.sqrt(mse)))
    print(f'r2_score - {r2}')

plt.close("all")
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(lag_range, scores)
ax1.set_xlabel('Alpha (lag range)')
ax1.set_ylabel('Beta (Score)')
ax1.set_title('Model Score vs Lag Value')
ax1.axis('tight')

ax2 = plt.gca()
ax2.ticklabel_format(useOffset=False)
errors = [round(num, 5) for num in errors]
ax2.plot(lag_range, errors)
ax2.set_xlabel("lag")
ax2.set_ylabel("error")
ax2.set_title("Error vs Lag")
ax2.axis("tight")
fig.show()
print(f'score for lasso model - {scores}')
print(f'error for lasso model - {errors}')

# CROSS VALIDATE and identify polynomial features
q = 2
lag = 3
stride = 1
yy, _, XX = utils.get_many_feature_data(bikes, time, q, lag, stride, dt, day_of_month, hour)
train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)

coeff = []
alphas = []
errors = []
scores = []
Ci = 10
qrange = [1, 2, 3, 4]
c_temp = []
mean_error = []
std_error = []
for q in qrange:
    print(q)
    temp = []
    XPoly = PolynomialFeatures(q).fit_transform(XX)
    a = 1 / (2 * Ci)
    print(Ci, a)
    model = Lasso(alpha=a, tol=1e-2).fit(XPoly[train], yy[train])
    coeff.append(model.coef_)
    alphas.append(a)
    y_pred = model.predict(XPoly)
    score = model.score(XPoly[test], yy[test])
    cross_val = cross_val_score(model, XPoly[test], yy[test], cv=10)
    mean_error.append(np.mean(cross_val))
    std_error.append(np.std(cross_val))
    mse = mean_squared_error(yy, y_pred)
    errors.append(np.sqrt(mse))
    r2 = r2_score(yy, y_pred)
    scores.append(r2)
    print("SCORE: {0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
          .format(score, mse, np.sqrt(mse)))
    print(f'r2_score - {r2}')

plt.close("all")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(qrange, scores)
ax1.set_xlabel('Alpha (Polynomial range)')
ax1.set_ylabel('Beta (Score)')
ax1.set_title('Model Score vs Polynomial Value')
ax1.axis('tight')

ax2.errorbar(qrange, mean_error, yerr=std_error)
ax2.set_xticks(qrange)
ax2.set_xlabel("ploynomial value")
ax2.set_ylabel("error")
ax2.set_title("Error vs polynomial")
ax2.axis("tight")

ax3 = plt.gca()
ax3.ticklabel_format(useOffset=False)
errors = [round(num, 5) for num in errors]
ax3.plot(qrange, errors)
ax3.set_xlabel("ploynomial value")
ax3.set_ylabel("error")
ax3.set_title("Error vs polynomial")
ax3.axis("tight")
fig.show()
plt.show()

# find out C with cross validation
q = 2
lag = 3
stride = 1
yy, tt, XX = utils.get_many_feature_data(bikes, time, q, lag, stride, dt, day_of_month, hour)
train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
coeff = []
alphas = []
errors = []
c_mean_error = []
c_std_error = []
C = [0.01, 0.1, 1, 3, 5, 10, 50, 100]

for Ci in C:
    c_temp = []
    mean_error = []
    std_error = []
    XPoly = PolynomialFeatures(4).fit_transform(XX)
    a = 1 / (2 * Ci)
    print(Ci, a)
    model = Lasso(alpha=a).fit(XPoly[train], yy[train])
    print("coefficient ", model.intercept_, model.coef_)
    coeff.append(model.coef_)
    alphas.append(a)
    y_pred = model.predict(XPoly)
    score = model.score(XPoly[test], yy[test])
    cross_val = cross_val_score(model, XPoly[test], yy[test], cv=10)
    c_mean_error.append(np.mean(cross_val))
    c_std_error.append(np.std(cross_val))
    mse = mean_squared_error(yy, y_pred)
    errors.append(np.sqrt(mse))
    print("SCORE: {0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}".format(score, mse, np.sqrt(mse)))
    print(f'r2_score - {r2_score(yy, y_pred)}')

plt.close("all")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(alphas, coeff)
ax1.set_xlabel('Alpha (Regularization Parameter)')
ax1.set_ylabel('Beta (Predictor Coefficients)')
ax1.set_title('Ridge Coefficients vs Regularization Parameters')
ax1.axis('tight')

ax2.errorbar(C, c_mean_error, yerr=c_std_error)
ax2.set_xlabel("polynomial value")
ax2.set_ylabel("error")
ax2.set_title("Error vs polynomial")
ax2.axis("tight")

ax3 = plt.gca()
ax3.ticklabel_format(useOffset=False)
errors = [round(num, 5) for num in errors]
ax3.plot(alphas, errors)
ax3.set_xlabel("alpha")
ax3.set_ylabel("RMSE error")
ax3.set_title("Coefficient error as a function of the regularization")
ax3.axis("tight")
fig.show()

# final run with all identified parameters
t_full = pd.array(pd.DatetimeIndex(suburb_df.iloc[:, 1]).astype(np.int64)) / 1000000000
dt = t_full[1] - t_full[0]
qrange = [2, 6, 12]
lag = 3
stride = 1
plt.close("all")
fig, ax = plt.subplots(3)
ax_index = 0
Ci = 50
poly = 4
q_score = {}
xlim = -5
ylim = 35  # 40 for Grangegorman
for q in qrange:
    yy, tt, XX = utils.get_many_feature_data(bikes, time, q, lag, stride, dt, day_of_month, hour)
    train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
    XPoly = PolynomialFeatures(poly).fit_transform(XX)
    a = 1 / (2 * Ci)
    print(f'Alpha & Ci  Value - {a} {Ci}')
    print(f"q value - {q}")
    model = Lasso(alpha=a).fit(XPoly[train], yy[train])
    y_pred = model.predict(XPoly)
    score = model.score(XPoly[test], yy[test])
    mse = mean_squared_error(yy, y_pred)
    print("SCORE: {0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}".format(score, mse, np.sqrt(mse)))
    r2 = r2_score(yy, y_pred)
    print(f'r2_score - {r2}\n')
    q_score[q] = r2

    ax[ax_index].scatter(time, bikes, color="blue")
    ax[ax_index].scatter(tt, y_pred, color="green")
    ax[ax_index].set_xlabel("time (days)")
    ax[ax_index].set_ylabel("#bikes")
    ax[ax_index].set_title(f"Merrion Square - Lasso model for q = {q}")
    ax[ax_index].legend(["training data", "predictions"], loc="upper right")
    ax[ax_index].set_ylim([xlim, ylim])
    ax_index += 1
print(({"-" * 30}))
for q, score in q_score.items():
    print(f"q value {q} - score {score}")
fig.show()
