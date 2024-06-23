"""
file: model_optimzation.py
---

This file accepts: 
- the path to a .pkl for a HorseRaceDataSet class
- parameters to specify a model for predicting performance:
    - X_cat_names: category names for the independent variables {"composition", "size", "task_attr", "task_complexity", "comms"}
    - y_name: name of the dependent variable
    - model_type: {RandomForestRegressor, ElasticNet}
- parameters for optimization using Ax: e.g., total_trials for BO

This file generates: a HorseRaceOptimizedModel containing the optimal parameters, according to Ax

"""

# 3rd Party Imports
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import pickle
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import ast 
from tqdm import tqdm
from copy import deepcopy
import itertools
import os
import warnings
warnings.filterwarnings('ignore') # outputs are too noisy with warnings re: non-convergence

from ax import optimize
from ax.plot.contour import plot_contour
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.trace import optimization_trace_single_method
import json
import torch
mps_device = torch.device("mps")

# Imports from my own library
from model_eval_setup.model_dict import model_dict

class HorseRaceOptimizedModel:
    def __init__(
        self,
        HorseRaceDataSet_path: str,
        X_cat_names: list,
        y_name: str,
        model_type: str,
        total_trials: int,
        inversely_weight_tasks = True,
        handle_na_values = "mean",
        na_fill_value = None	
    ) -> None:

        """
        Parameters:
        @HorseRaceDataSet_path: a path to a .pkl of a HorseRaceDataSet
        @X_cat_names: the category names of the desired X's: one of {"composition", "size", "task_attr", "task_complexity", "comms"}
        @y_name: the desired dependent variable
        @model_type: a string specifying the model. One of {RandomForestRegressor, ElasticNet}
        @total_trials: number of trials for optimizing the model
        @inversely_weight_tasks (Defaults to True): whether to weight datapoints from different tasks inversely depending on how frequently the type of task appeared in the data.
        @handle_na_values (Defaults to "mean"): determines how to handle missing values; accepted values are {"drop", "mean", "median", "most_frequent", or "constant"}. 
            If "constant," must also provide a fill value (`na_fill_value`)
        @na_fill_value: the value with which the user specifies filling NA values. 
        """
        self.total_trials = total_trials
        self.inversely_weight_tasks = inversely_weight_tasks

        # Assert that the model name is valid entry in the dictionary of what we are exploring
        self.model_dict = model_dict
        assert model_type in self.model_dict.keys(), "Model type not supported. Currently supported: " + str(self.model_dict.keys())
        self.model_type = model_type

        # Assert that the na handling options are valid
        assert handle_na_values in {"drop", "mean", "median", "most_frequent", "constant"}, "Invalid specification of handling NA values."
        if handle_na_values == "drop":
            self.drop_na = True
            self.imputation_option = None
        else:
            self.drop_na = False
            self.imputation_option = handle_na_values

        self.na_fill_value = na_fill_value

        # First, unpack the HorseRaceDataSet object and get the X's and y's
        with open(HorseRaceDataSet_path, "rb") as horseracedataset_file:
            self.HorseRaceData = pickle.load(horseracedataset_file)

        # Assert that the desired y's are valid
        assert y_name in set(self.HorseRaceData.dvs.columns), "Dependent variable (y) not found."
        # Set the y
        self.y_name = y_name
        self.y = self.HorseRaceData.dvs[y_name]

        # Assert that the desired X's are valid
        assert set(X_cat_names) <= {"composition", "size", "task_attr", "task_complexity", "comms"}, "Unsupported independent variable detected."

        # set the X's
        X_list = []

        if "composition" in X_cat_names:
            X_list.append(self.HorseRaceData.composition_features)
        if "size" in X_cat_names:
            X_list.append(self.HorseRaceData.size_feature)
        if "task_attr" in X_cat_names:
            X_list.append(self.HorseRaceData.task_features)
        if "task_complexity" in X_cat_names:
            X_list.append(self.HorseRaceData.task_complexity_features)
        if "comms" in X_cat_names:
            X_list.append(self.HorseRaceData.conversation_features)
        
        # Append the different columns selected into the X's
        self.X = pd.concat(X_list, axis=1)

        # Handle NA's according to the specs
        self.handle_na_values()

        # Get and save the optimal model parameters
        # add a column called "task_name", as this is required for the leave-a-task-out Q^2
        y_task_appended = self.y.copy()
        y_task_appended["task_name"] = self.HorseRaceData.task_name

        X_task_appended = self.X.copy()
        X_task_appended["task_name"] = self.HorseRaceData.task_name
        self.get_optimal_model(X_task_appended, y_task_appended)

    """
    Handle NA values according to the user's specifications.
    """
    def handle_na_values(self) -> None:
        data = pd.concat([self.X.reset_index(drop=True), self.y.reset_index(drop=True)], axis=1)
        
        if self.drop_na:
            data = data.dropna()
            self.X = pd.DataFrame(data[:, :-1])
            self.y = pd.DataFrame(data[:, -1])
        else:  # apply an imputation option
            data = data.to_numpy()
            if self.imputation_option != "constant":
                imputer = SimpleImputer(strategy=self.imputation_option)
                data = imputer.fit_transform(data)
            else:
                assert self.na_fill_value is not None, "You cannot specify a 'constant' NA fill option without specifying an `na_fill_value.`"
                imputer = SimpleImputer(strategy=self.imputation_option, fill_value=self.na_fill_value)
                data = imputer.fit_transform(data)		
            self.X = pd.DataFrame(data[:, :-1], columns = self.X.columns)
            self.y = pd.DataFrame(data[:, -1], columns = [self.y_name])

        assert (self.X.isna().sum() == 0).all(), "NA's found in X."
        assert (self.y.isna().sum() == 0).all(), "NA's found in y."

    """
    Since we have an imbalance in the number of samples we have from different tasks, we need to inversely
    weight the data based on the task.

    TODO -- this functionality hasn't been recreated because it was based on the previous leave-one-instance model
    """
    # def get_sample_weighting(self, task_names):
    # 	task_proportions = task_names.value_counts()/len(task_names)
    # 	return(task_names.apply(lambda task: 1 / task_proportions[task]))
    
    # This function computes q^2 (used as evaluation for the models)
    # Q^2
    def reshape_x_y(self, x, y):
        if(isinstance(x, pd.Series)):
            x = np.asarray(x).reshape(-1, 1)
        else:
            x = np.asarray(x)
        
        y = np.asarray(y).reshape(-1, 1)
        return(x, y)

    def q2_task_holdout_helper(self, x_train, x_test, y_train, y_test, estimator):
        
        # some reshaping
        x_train_array, y_train_array = self.reshape_x_y(x_train, y_train)
        x_test_array, y_test_array = self.reshape_x_y(x_test, y_test)

        # Fit the model and get the error
        fitted_model = estimator.fit(X=x_train_array, y=y_train_array.ravel())
        
        # save prediction error
        prediction = fitted_model.predict(x_test_array)

        # flatten all arrays
        y_test_array = np.asarray(y_test_array).flatten()
        prediction = np.asarray(prediction).flatten()

        squared_model_prediction_error = (y_test_array - prediction) ** 2

        # save total error for this fold
        squared_average_prediction_error = (y_test_array - np.mean(y_train_array)) ** 2

        return squared_model_prediction_error, squared_average_prediction_error

    """
    This is the version of q^2 that holds out EVERYTHING associated with a given task

    It trains on all task instances from the "seen" classes, and it tests on task instances of held-out (unseen) classes.

    NOTE: this version of the function assumes that x and y are passed in with a column called "task_name"
    """

    def get_q2(self, y, x, estimator = Lasso(), num_task_holdouts = 1):

        squared_model_prediction_errors = []
        squared_average_prediction_errors = []

        num_total_tasks = x["task_name"].nunique()

        # randomly hold out `num_task_holdouts`
        all_possible_task_combos = list(itertools.combinations((x["task_name"].unique()), num_total_tasks - num_task_holdouts))
        
        for sample in all_possible_task_combos:

            x_train_tasks = x[x["task_name"].isin(sample)].drop("task_name", axis = 1)
            x_test_tasks = x[~x["task_name"].isin(sample)].drop("task_name", axis = 1)

            y_train_tasks = y[y["task_name"].isin(sample)].drop("task_name", axis = 1)
            y_test_tasks = y[~y["task_name"].isin(sample)].drop("task_name", axis = 1)

            # get evaluation score by training on the training tasks and evaluating on the holdout tasks
            squared_model_prediction_error, squared_average_prediction_error = self.q2_task_holdout_helper(x_train_tasks, x_test_tasks, y_train_tasks, y_test_tasks, estimator)
            
            squared_model_prediction_errors.append(squared_model_prediction_error)
            squared_average_prediction_errors.append(squared_average_prediction_error)

        squared_model_prediction_error = np.asarray(squared_model_prediction_error).flatten()
        squared_average_prediction_error = np.asarray(squared_average_prediction_error).flatten()

        return 1 - (np.sum(squared_model_prediction_error) / np.sum(squared_average_prediction_error))


    def get_optimal_model(self, X_task_appended, y_task_appended):

        best_parameters, best_values, experiment, model = optimize(
            parameters=self.model_dict[self.model_type]["params"],
            evaluation_function=lambda p: self.get_q2(y_task_appended, X_task_appended, estimator=self.model_dict[self.model_type]["model"](**p)),
            minimize=False,
            total_trials=self.total_trials,
        )
        self.optimal_parameters = best_parameters 
        self.optimization_q2 = best_values[0]["objective"]