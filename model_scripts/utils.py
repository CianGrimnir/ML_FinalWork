import numpy as np
import math


# return many-features data for dublin bikes dataset (bike occupancy, day of the month, and hour)
def get_many_feature_data(bikes, time, q, lag, stride, dt, day_of_month, hour):
    w = math.floor(7 * 24 * 60 * 60 / dt)
    length = bikes.size - w - lag * w - q
    print("initial set : ", bikes.size, w, lag, q, length)
    XX = bikes[q:q + length:stride]
    X1 = day_of_month[q:q + length:stride]
    X2 = hour[q:q + length:stride]
    XX = np.column_stack((XX, X1, X2))
    # week
    for i in range(1, lag):
        X = bikes[i * w + q:i * w + q + length:stride]
        X1 = day_of_month[i * w + q:i * w + q + length:stride]
        X2 = hour[i * w + q:i * w + q + length:stride]
        XX = np.column_stack((XX, X, X1, X2))
    d = math.floor(24 * 60 * 60 / dt)
    # days
    for i in range(0, lag):
        X = bikes[i * d + q:i * d + q + length:stride]
        X1 = day_of_month[i * d + q:i * d + q + length:stride]
        X2 = hour[i * d + q:i * d + q + length:stride]
        XX = np.column_stack((XX, X, X1, X2))

    for i in range(0, lag):
        X = bikes[i:i + length:stride]
        X1 = day_of_month[i:i + length:stride]
        X2 = hour[i:i + length:stride]
        XX = np.column_stack((XX, X, X1, X2))

    yy = bikes[lag * w + w + q:lag * w + w + q + length:stride]
    tt = time[lag * w + w + q:lag * w + w + q + length:stride]

    yy.reset_index(drop=True, inplace=True)
    tt.reset_index(drop=True, inplace=True)

    return yy, tt, XX


# return single feature (bike occupancy) for dublin bikes dataset
def get_feature_data(bikes, time, lag, q, stride, dt):
    w = math.floor(7 * 24 * 60 * 60 / dt)
    length = bikes.size - w - lag * w - q
    print("initial set : ", bikes.size, w, lag, q, length)
    XX = bikes[q:q + length:stride]
    # week
    for i in range(1, lag):
        X = bikes[i * w + q:i * w + q + length:stride]
        XX = np.column_stack((XX, X))
    d = math.floor(24 * 60 * 60 / dt)
    # days
    for i in range(0, lag):
        X = bikes[i * d + q:i * d + q + length:stride]
        XX = np.column_stack((XX, X))

    for i in range(0, lag):
        X = bikes[i:i + length:stride]
        XX = np.column_stack((XX, X))

    yy = bikes[lag * w + w + q:lag * w + w + q + length:stride]
    tt = time[lag * w + w + q:lag * w + w + q + length:stride]

    yy.reset_index(drop=True, inplace=True)
    tt.reset_index(drop=True, inplace=True)

    return yy, tt, XX
