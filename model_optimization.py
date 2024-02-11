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
		self.get_optimal_model(self.X, self.y, self.HorseRaceData.task_name)

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

		q2 = 1 - np.sum((np.array(q2_preds) - np.array(y))**2) / np.sum((np.array(q2_means) - np.array(y))**2)
		
		return q2

	def get_optimal_model(self, X, y, task_names):

		best_parameters, best_values, experiment, model = optimize(
			parameters=self.model_dict[self.model_type]["params"],
			evaluation_function=lambda p: self.q2_score(self.model_dict[self.model_type]["model"](**p), X, y, task_names),
			minimize=False,
			total_trials=self.total_trials,
		)
		self.optimal_parameters = best_parameters 
		self.optimization_q2 = best_values[0]["objective"]