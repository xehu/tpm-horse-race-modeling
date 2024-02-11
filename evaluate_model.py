"""
file: evaluate_model.py
---
This file is the main driver of the model evaluation pipeline.

"""

import pandas as pd
import numpy as np
import os, glob
import sys
from model_optimization import HorseRaceOptimizedModel
from model_evaluation import HorseRaceModelEvaluator
from importlib import import_module
import yaml
from model_eval_config.model_eval_config_schema import model_eval_config_schema as CONFIG_SCHEMA
import argparse
import pickle
import concurrent.futures
clean_multi_task_data = import_module('tpm-data.data_cleaning.clean_multi_task_data').clean_multi_task_data

def read_and_validate_config(user_provided_config):
	with open(user_provided_config, 'r') as file:
		config_data = yaml.safe_load(file)

	# Validate the yaml schema
	for section, schema in CONFIG_SCHEMA.items():
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

def get_optimal_model(model_paths, model_options, optimization_options, optimization_output_filepath):

	print("Getting optimal model params...")
	# Run the Model Evaluation
	optimized_model = HorseRaceOptimizedModel(
		HorseRaceDataSet_path=model_paths["HorseRaceDataSet_path"],
		X_cat_names=model_options["X_cat_names"],
		y_name=model_options["y_name"],
		model_type=model_options["model_type"],
		total_trials=optimization_options["total_trials"],
		inversely_weight_tasks=model_options["inversely_weight_tasks"],
		handle_na_values=model_options["handle_na_values"],
		na_fill_value=model_options["na_fill_value"]		
	)

	# Save it as a pickle
	os.makedirs(os.path.dirname(optimization_output_filepath), exist_ok=True) # make the path if it doesn't exist
	with open(optimization_output_filepath, "wb") as pickle_file:
		pickle.dump(optimized_model, pickle_file)

def evaluate_model(config_file):
	
	model_paths, model_options, optimization_options = read_and_validate_config(config_file)

	# If optimal model parameters do not yet exist, generate them.
	# automatically name the file in the output directory, with the categories used, the model type, dv, and total trials
	optimization_output_filepath = model_paths["optimization_output_path"] + "/" + '_'.join(model_options["X_cat_names"]) + '_' + model_options["model_type"] + "_" + model_options["y_name"] + "_" + str(optimization_options["total_trials"])

	print("Checking for optimal model params...")
	if not os.path.isfile(optimization_output_filepath):
		get_optimal_model(model_paths, model_options, optimization_options, optimization_output_filepath)

	# Run the Model Evaluation based on the optimal parameters that we found.
	print("Evaluating model...")
	model_evaluator = HorseRaceModelEvaluator(
		HorseRaceOptimizedModel_path = optimization_output_filepath,
		n_iterations=optimization_options["n_iterations"],
		output_file_path = model_paths["evaluation_output_path"],
		output_file_mode =optimization_options["output_file_mode"]
	)
	model_evaluator.evaluate_optimal_model()
	print("All Done!")

# Main Function
if __name__ == "__main__":

	# Parse command-line arguments
	parser = argparse.ArgumentParser(description="This program takes in a config file for the TPM Horse Race project. For a specified HorseRaceDataSet and model, fits and evaluates the model with optimal parameters.")
	parser.add_argument('--evaluate', nargs=1, help='Run the model evaluation sequence according to [config]. You need to pass in 1 argument, [config], which should be a path pointing to the config file.')
	parser.add_argument('--multieval', nargs=1, help='Run the model evaluation sequence according to each config file within a directory, passed in as the argument. You need to pass in 1 argument, [directory], which should be a path pointing to the directory containing all config files.')
	args = parser.parse_args()

	if args.evaluate:
		# Evaluate just one config file
		config_file_path = args.evaluate[0]
		evaluate_model(config_file_path)
	
	elif args.multieval:
		# Evaluate multiple config files at once
		path = args.multieval[0]
		config_files = glob.glob(os.path.join(path, '*.yaml'))

		with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
			futures = [executor.submit(evaluate_model, config_file) for config_file in config_files]
			
	else:
		print("No arguments provided. Usage: --evaluate [config] or --multieval [directory with configs]")