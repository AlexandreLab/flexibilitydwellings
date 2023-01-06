import numpy as np


def MAE(actual_values, predicted_values):
    return np.sum(np.abs(actual_values - predicted_values)) / len(actual_values)


def MAPE(actual_values, predicted_values):
    return np.sum(np.abs((actual_values - predicted_values) / actual_values)) / len(
        actual_values
    )


def RMSE(actual_values, predicted_values):
    print(actual_values, predicted_values)
    MSE = np.sum((actual_values - predicted_values) ** 2) / len(actual_values)
    return np.sqrt(MSE)
