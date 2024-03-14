import concurrent.futures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

aggregated_performance_by_task = pd.read_csv('./aggregated_performance_by_task.csv')

def get_q2_concurrent(y, x):
    squared_model_prediction_errors = []
    squared_average_prediction_errors = []

    # Function to calculate prediction errors for a single iteration
    def calculate_errors(i):
        y_test = y.iloc[[i]]
        x_test = x.iloc[[i]]
        y_train = y.drop(index=y.index[i]).reset_index(drop=True)
        x_train = x.drop(index=x.index[i]).reset_index(drop=True)

        if len(x_train) == 0:
            # Skip iteration if the training set is empty after dropping the row
            return 0, 0

        x_train_array = np.asarray(x_train)
        if len(x.shape) == 1:
            x_train_array = x_train_array.reshape(-1, 1)
        y_train_array = np.asarray(y_train).reshape(-1, 1)

        x_test_array = np.asarray(x_test)
        if len(x.shape) == 1:
            x_test_array = x_test_array.reshape(-1, 1)  # Ensure x_test is 2D
        y_test_array = np.asarray(y_test).reshape(-1, 1)

        fitted_model = LinearRegression().fit(X=x_train_array, y=y_train_array)

        # Calculate prediction error
        prediction = fitted_model.predict(x_test_array)[0]
        model_error = (y_test_array - prediction) ** 2

        # Calculate total error for this fold
        average_error = (y_test_array - np.mean(y_train_array)) ** 2

        return model_error, average_error

    # Parallelize the loop using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as q2_executor:
        results = list(q2_executor.map(calculate_errors, range(len(y))))

    # Unpack the results
    squared_model_prediction_errors, squared_average_prediction_errors = zip(*results)

    return 1 - (np.sum(squared_model_prediction_errors) / np.sum(squared_average_prediction_errors))

def calculate_q2_for_combination(params):
    dv, combination = params # unpack param
    return get_q2_concurrent(aggregated_performance_by_task[dv], aggregated_performance_by_task[list(combination)])