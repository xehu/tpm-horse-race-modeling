"""
file: evaluate_model.py
---
This file is the main driver of the model evaluation pipeline.

"""

import pandas as pd
import numpy as np
import os
import sys
from model_evaluation import HorseRaceModelEvaluator
from importlib import import_module
import yaml
from model_eval_config.model_eval_config_schema import model_eval_config_schema
import argparse
import pickle
clean_multi_task_data = import_module('tpm-data.data_cleaning.clean_multi_task_data').clean_multi_task_data

def read_and_validate_config(model_eval_config_schema, user_provided_config):
	with open(user_provided_config, 'r') as file:
		config_data = yaml.safe_load(file)

	# Validate the yaml schema
	for section, schema in model_eval_config_schema.items():
		if section not in config_data:
			config_data[section] = {}  # Create empty section if not provided

		for key, default_value in schema.items():
			if key not in config_data[section]:
				config_data[section][key] = default_value  # Replace missing values with defaults

	# Read each element into a variable
	model_paths = config_data.get("model_paths", {})
	model_options = config_data.get("model_options", {})
	optimization_options = config_data.get("optimization_options", {})

	return model_paths, model_options, optimization_options


# Main Function
if __name__ == "__main__":

	# Parse command-line arguments
	parser = argparse.ArgumentParser(description="This program takes in a config file for the TPM Horse Race project. For a specified HorseRaceDataSet and model, fits and evaluates the model with optimal parameters.")
	parser.add_argument('--evaluate', nargs=1, help='Run the model evaluation sequence according to [config]. You need to pass in 1 argument, [config], which should be a path pointing to the config file.')
	args = parser.parse_args()

	if args.evaluate:
		config_file_path = args.evaluate[0]
		model_paths, model_options, optimization_options = read_and_validate_config(model_eval_config_schema, config_file_path)

		# Run the Model Evaluation
		model_evaluator = HorseRaceModelEvaluator(
			HorseRaceDataSet_path = model_paths["HorseRaceDataSet_path"],
			X_cat_names = model_options["X_cat_names"],
			y_name = model_options["y_name"],
			model_type = model_options["model_type"],
			optimization_output_path = model_paths["optimization_output_path"],
			n_iterations = optimization_options["n_iterations"],
			total_trials = optimization_options["total_trials"],
			inversely_weight_tasks = model_options["inversely_weight_tasks"],
			handle_na_values = model_options["handle_na_values"],
			na_fill_value = model_options["na_fill_value"]
			)
		model_evaluator.evaluate_optimal_model()

	else:
		print("No arguments provided. Usage: --evaluate [config]")