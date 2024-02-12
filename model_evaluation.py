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
	"""
	def get_sample_weighting(self, task_names):
		task_proportions = task_names.value_counts()/len(task_names)
		return(task_names.apply(lambda task: 1 / task_proportions[task]))

	"""
	Credit for q2_baseline_models and q2_score: Mohammed Alsobay
	"""
	def q2_baseline_models(self, estimator, X, y, task_names):
		baseline_models = []

		sample_weights = None
		if(self.HorseRaceOptimizedModel.inversely_weight_tasks):
				sample_weights = self.get_sample_weighting(task_names)
		
		for loo_index in self.HorseRaceOptimizedModel.X.index:
			if hasattr(estimator, "random_state"):
				estimator.random_state = loo_index	
			if (self.HorseRaceOptimizedModel.inversely_weight_tasks):
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

		# Get the q^2 of model
		return self.q2_score(estimator, X, y, task_names)

	def evaluate_optimal_model(self):
		os.makedirs(os.path.dirname(self.final_output_path), exist_ok=True)
		output = open(self.final_output_path, self.output_file_mode)

		for iteration in range(self.n_iterations):
			sample_size = len(self.HorseRaceOptimizedModel.X.index)
			data = pd.concat([self.HorseRaceOptimizedModel.X.reset_index(drop=True), self.HorseRaceOptimizedModel.y.reset_index(drop=True), self.HorseRaceOptimizedModel.HorseRaceData.task_name.reset_index(drop=True)], axis = 1)
			data_resampled = data.sample(n = sample_size, random_state = iteration).reset_index(drop=True)

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
