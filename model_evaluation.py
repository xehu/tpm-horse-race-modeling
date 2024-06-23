"""
file: model_evaluation.py
----

This file accepts: 
- the path to a .pkl for a HorseRaceOptimizedModel, which specifies the model and associated parameters.

This file outputs:
- An evaluation for the model (in Q^2).
	- Each line of the file is the results of one iteration;
		we iterate through `n_iterations`, with a different bootstrapped re-sampling each time.
"""

# 3rd Party Imports
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import pickle
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import ast 
from tqdm import tqdm
from copy import deepcopy
import itertools
import os
import threading
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

class HorseRaceModelEvaluator:
	def __init__(
		self,
		HorseRaceOptimizedModel_path: str,
		n_iterations: int,
		output_file_path: str,
		output_file_mode = 'a'
	) -> None:

		"""
		Parameters:
		@HorseRaceOptimizedModel_path: a path to a .pkl of a HorseRaceOptimizedModel
		@n_iterations: number of times to evaluate the model
		@output_file_path: a path for the directory where the model evaluation outputs are to be written.
		@output_file_mode: One of append ('a') or write ('w'); controls whether we append to the output file or simply overwrite it.
		"""
		
		# automatically name the file in the output directory
		self.final_output_path = output_file_path + '/' + HorseRaceOptimizedModel_path.split("/")[-1] + "_evaluation"
		self.n_iterations = n_iterations	

		assert output_file_mode in {'a', 'w'}, "Only two output file modes accepted: append('a') and write ('w')."
		self.output_file_mode = output_file_mode

		# Unpack the pickled HorseRaceOptimizedModel object
		with open(HorseRaceOptimizedModel_path, "rb") as horseraceoptimizedmodel_file:
			self.HorseRaceOptimizedModel = pickle.load(horseraceoptimizedmodel_file)

		# The Model Dict
		self.model_dict = model_dict

		# Thread locker for multithreading
		self.lock = threading.Lock()

	"""
	Since we have an imbalance in the number of samples we have from different tasks, we need to inversely
	weight the data based on the task.

	TODO -- this functionality hasn't been recreated because it was based on the previous leave-one-instance model
	"""
	# def get_sample_weighting(self, task_names):
	# 	task_proportions = task_names.value_counts()/len(task_names)
	# 	return(task_names.apply(lambda task: 1 / task_proportions[task]))

	"""
	Q^2 functions for evalutation
	"""
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

	# Fit a model based on the parameters
	def call_model(self, X, y, task_names):
		optimal_params = self.HorseRaceOptimizedModel.optimal_parameters

		# Set up model with the optimal parameters
		estimator = self.model_dict[self.HorseRaceOptimizedModel.model_type]["model"]()

		# params to fill
		params_to_fill = self.model_dict[self.HorseRaceOptimizedModel.model_type]["params"]

		for param in params_to_fill:
			param_name = param["name"]
			param_value = optimal_params[param_name]
			estimator.set_params(**{param_name: param_value})

		# attach the task_names to y and X before putting into the q^2 function
		# add a column called "task_name", as this is required for the leave-a-task-out Q^2
		y_task_appended = pd.DataFrame(y.copy())
		y_task_appended["task_name"] = task_names

		X_task_appended = X.copy()
		X_task_appended["task_name"] = task_names

		# Get the q^2 of model
		return self.get_q2(y_task_appended, X_task_appended, estimator)

	def evaluate_optimal_model(self):
		os.makedirs(os.path.dirname(self.final_output_path), exist_ok=True)
		output = open(self.final_output_path, self.output_file_mode)

		for iteration in range(self.n_iterations):
			sample_size = len(self.HorseRaceOptimizedModel.X.index)
			data = pd.concat([self.HorseRaceOptimizedModel.X.reset_index(drop=True), self.HorseRaceOptimizedModel.y.reset_index(drop=True), self.HorseRaceOptimizedModel.HorseRaceData.task_name.reset_index(drop=True)], axis = 1)
			data_resampled = data.sample(n = sample_size, random_state = iteration, replace = True).reset_index(drop=True)

			# assert that the resampling worked
			assert data.shape == data_resampled.shape, "Resampled dataframe does not have the correct shape. Stopping..."
			assert not data_resampled.equals(data), "Resampled data is exactly the same as data. Stopping ..."
			
			X_resampled = data_resampled[self.HorseRaceOptimizedModel.X.columns]
			y_resampled = data_resampled[self.HorseRaceOptimizedModel.y_name]
			task_names_resampled = data_resampled[self.HorseRaceOptimizedModel.HorseRaceData.task_name.name]
			
			self.lock.acquire() # Lock the evaluation & writing of one file

			evaluation_output = {}
			q2 = self.call_model(X_resampled, y_resampled, task_names_resampled)

			evaluation_output["q^2"] = q2
			evaluation_output["iteration"] = iteration

			output.write(str(evaluation_output) + "\n")
			output.flush()

			self.lock.release()
