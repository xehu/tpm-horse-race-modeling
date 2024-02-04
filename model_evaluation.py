"""
file: model_evaluation.py
----

This file accepts: 
- the path to a .pkl for a HorseRaceDataSet class
- parameters to specify a model for predicting performance:
	- X_cat_names: category names for the independent variables {"composition", "size", "task_attr", "task_complexity", "comms"}
	- y_name: name of the dependent variable
	- model_type: {RandomForestRegressor, ElasticNet}

This file outputs:
- An evaluation for the model (in Q^2), alongside the different optimal search space values for each iteration.
	- Each line of the file is the results of one iteration;
		we iterate through `n_iterations`, with a different bootstrapped re-sampling each time.
	- Within each iteration, we get the optimized parameters for *that resampling* and save the results.
	- We use `total_trials` to determine the number of times we iterate (using the Bayesian Optimization method)
		to get the best parameters in each trial.
"""

# 3rd Party Imports
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import pickle
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import ast 
from tqdm import tqdm
from copy import deepcopy
import itertools

from ax import optimize
from ax.plot.contour import plot_contour
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.trace import optimization_trace_single_method
import json

# Imports from my own library
from model_eval_setup.model_dict import model_dict

class HorseRaceModelEvaluator:
	def __init__(
		self,
		HorseRaceDataSet_path: str,
		X_cat_names: list,
		y_name: str,
		model_type: str,
		optimization_output_path: str,
		n_iterations: int,
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
		@optimization_output_path: a path for where the model evaluation outputs are to be written.
		@n_iterations: number of times to evaluate the model
		@total_trials: number of trials for optimizing the model
		@inversely_weight_tasks (Defaults to True): whether to weight datapoints from different tasks inversely depending on how frequently the type of task appeared in the data.
		@handle_na_values (Defaults to "mean"): determines how to handle missing values; accepted values are {"drop", "mean", "median", "most_frequent", or "constant"}. 
			If "constant," must also provide a fill value (`na_fill_value`)
		@na_fill_value: the value with which the user specifies filling NA values. 
		"""
		self.optimization_output_path = optimization_output_path
		self.n_iterations = n_iterations
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

	"""
	Handle NA values according to the user's specifications.
	"""
	def handle_na_values(self) -> None:
		data = pd.concat([self.X.reset_index(drop=True), self.y.reset_index(drop=True)], axis=1).to_numpy()

		if self.drop_na:
			data = data.dropna()
		else:  # apply an imputation option
			if self.imputation_option != "constant":
				imputer = SimpleImputer(strategy=self.imputation_option)
				data = imputer.fit_transform(data)
			else:
				assert self.na_fill_value is not None, "You cannot specify a 'constant' NA fill option without specifying an `na_fill_value.`"
				imputer = SimpleImputer(strategy=self.imputation_option, fill_value=self.na_fill_value)
				data = imputer.fit_transform(data)

		self.X = pd.DataFrame(data[:, :-1], columns = self.X.columns)
		self.y = pd.DataFrame(data[:, -1], columns = [self.y_name])

	"""
	Since we have an imbalance in the number of samples we have from different tasks, we need to inversely
	weight the data based on the task.
	"""
	def get_sample_weighting(self, task_names):
		task_proportions = task_names.value_counts()/len(task_names)
		return(task_names.apply(lambda task: 1 / task_proportions[task]))

	"""
	Credit for q2_baseline_models, q2_score, and ax_param_optimzation: Mohammed Alsobay
	----
	I used Mohammed's code as a basis for creating the Ax-based optimization for these models.
	"""
	def q2_baseline_models(self, estimator, X, y, task_names):
		baseline_models = []

		sample_weights = None
		if(self.inversely_weight_tasks):
				sample_weights = self.get_sample_weighting(task_names)
		
		for loo_index in self.X.index:
			if hasattr(estimator, "random_state"):
				estimator.random_state = loo_index	
			if (self.inversely_weight_tasks):
				sample_weight_loo = sample_weights.drop(loo_index)
			else:
				sample_weight_loo = None

			baseline_models.append(deepcopy(estimator.fit(X=X.drop(loo_index).values, y=y.drop(loo_index).values, sample_weight = sample_weight_loo)))
			
		return baseline_models
	
	# This function computes q^2 (used as evaluation for the models)	
	def q2_score(self, estimator, X, y, task_names):
		models = self.q2_baseline_models(estimator, X, y, task_names)
		q2_means = []
		q2_preds = []
		
		for loo_index in X.index:
			q2_means.append(y.drop(loo_index).mean())
			q2_preds.append(models[loo_index].predict(np.array(X.iloc[loo_index]).reshape(1,-1))[0])
		
		q2 = 1 - np.sum((np.array(q2_preds) - y)**2) / np.sum((np.array(q2_means) - y)**2)
		
		return q2

	def ax_param_optimization(self, X, y, task_names):
		best_parameters, best_values, experiment, model = optimize(
			parameters=self.model_dict[self.model_type]["params"],
			evaluation_function=lambda p: self.q2_score(self.model_dict[self.model_type]["model"](**p), X, y, task_names),
			minimize=False,
			total_trials=self.total_trials
		)
		ax_output = best_parameters 
		ax_output["q2"] = best_values[0]["objective"]
		ax_output["model_type"] = self.model_type

		return ax_output

	def evaluate_optimal_model(self):
		output = open(self.optimization_output_path, "w")

		for iteration in range(self.n_iterations):
			sample_size = len(self.X.index)
			data = pd.concat([self.X.reset_index(drop=True), self.y.reset_index(drop=True), self.HorseRaceData.task_name.reset_index(drop=True)], axis = 1)
			data_resampled = data.sample(n = sample_size, random_state = iteration).reset_index(drop=True)
			X_resampled = data_resampled[self.X.columns]
			y_resampled = data_resampled[self.y_name]
			task_names_resampled = data_resampled[self.HorseRaceData.task_name.name]
			
			try:
				ax_output = self.ax_param_optimization(X_resampled, y_resampled, task_names_resampled)
				ax_output["iteration"] = iteration
				output.write(str(ax_output) + "\n")
				output.flush()
			except Exception as e: 
				output.write(str(f"EXCEPTION WITH AX FOR {self.model_type} -- IN ITERATION #{iteration}") + "\n")
				output.write(str(e) + "\n")
				output.flush()
