"""
file: run_model.py
---
This file is the main driver of the model pipeline.

"""

import pandas as pd
import numpy as np
import os
import sys
from data import HorseRaceDataSet
from importlib import import_module
import yaml
from data_config.config_schema import config_schema
import argparse
import pickle

def read_and_validate_config(config_schema, config_test):
	# Read config_test.yaml
	with open(config_test, 'r') as file:
		config_data = yaml.safe_load(file)

	# Validate the schema
	for section, schema in config_schema.items():
		if section not in config_data:
			config_data[section] = {}  # Create empty section if not provided

		for key, default_value in schema.items():
			if key not in config_data[section]:
				config_data[section][key] = default_value  # Replace missing values with defaults

	# Read each element into a variable
	data_paths = config_data.get("data_paths", {})
	conv_featurizer_options = config_data.get("conv_featurizer_options", {})
	cleaning_options = config_data.get("cleaning_options", {})
	format_options = config_data.get("format_options", {})
	variable_options = config_data.get("variable_options", {})

	return data_paths, conv_featurizer_options, cleaning_options, format_options, variable_options


def call_FeatureBuilder(data_paths, conv_featurizer_options):
	sys.path.insert(0, './features/team-process-map/feature_engine/')
	FeatureBuilder = import_module("features.team-process-map.feature_engine.feature_builder").FeatureBuilder

	feature_builder = FeatureBuilder(
		# TODO --- update input file path once the dataset submodule is all set up
		# This will eventually point to the datasets folder
		input_file_path = data_paths["input"],
		output_file_path_chat_level = data_paths["output_chat"],
		output_file_path_user_level = data_paths["output_user"],
		output_file_path_conv_level = data_paths["output_conv"],
		analyze_first_pct = conv_featurizer_options["analyze_first_pct"],
		turns = conv_featurizer_options["turns"],
		conversation_id = conv_featurizer_options["conversation_id"],
		cumulative_grouping = conv_featurizer_options["cumulative_grouping"],
		within_task = conv_featurizer_options["within_task"],
	)
	feature_builder.featurize(col="message")

# Main Function
if __name__ == "__main__":

	# Parse command-line arguments
	parser = argparse.ArgumentParser(description="This program takes in a config file for the TPM Horse Race project and cleans the data appropriately.")
	parser.add_argument('--clean', nargs=1, help='Run the data cleaning sequence according to [config]. You need to pass in 1 argument, [config], which should be a path pointing to the config file.')
	args = parser.parse_args()

	if args.clean:
		config_file_path = args.clean[0]
		data_paths, conv_featurizer_options, cleaning_options, format_options, variable_options = read_and_validate_config(config_schema, config_file_path)

		# Raw Data Cleaning Stage
		# TODO --- set up the data cleaning in `tpm-data`
		# We should be able to run the data cleaning as a script here

		# Featurization Stage
		# If data is not featurizer, run the FeatureBuilder
		if not os.path.isfile(data_paths["output_conv"]):
			call_FeatureBuilder(data_paths, conv_featurizer_options)

		# Dataset assembly stage
		# Set up Dataset Class
		if not os.path.isfile(data_paths["output_pkl"]):
			dataset = HorseRaceDataSet(data_path =  data_paths["output_conv"],
				min_num_chats = cleaning_options["min_num_chats"],
				num_conversation_components = cleaning_options["num_conversation_components"],
				fillna_options = cleaning_options["fillna_options"],
				standardize_dv = cleaning_options["standardize_dv"],
				task_name_index = format_options["task_name_index"],
				complexity_name_index = format_options["complexity_name_index"],
				total_messages_varname = format_options["total_messages_varname"],
				dvs = variable_options["dvs"],
				composition_vars = variable_options["composition_vars"],
				task_vars = variable_options["task_vars"],
				task_name_mapping = variable_options["task_name_mapping"]
			)

			# Caching Stage
			with open(data_paths["output_pkl"], "wb") as pickle_file:
				pickle.dump(dataset, pickle_file)

	else:
		print("No arguments provided. Usage: --clean [config]")



