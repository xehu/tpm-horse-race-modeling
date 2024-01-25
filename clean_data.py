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

# Main Function
if __name__ == "__main__":
	
	# Run Data Cleaning
	# TODO --- set up the data cleaning in `tpm-data`
	# We should be able to run the data cleaning as a script here

	# Collect Features

		# Comms feature (call the featureBuilder)
		sys.path.insert(0, './features/team-process-map/feature_engine/')
		FeatureBuilder = import_module("features.team-process-map.feature_engine.feature_builder").FeatureBuilder

		feature_builder = FeatureBuilder(
			# TODO --- update input file path once the dataset submodule is all set up
			# This will eventually point to the datasets folder
			input_file_path = "./features/team-process-map/feature_engine/data/raw_data/multi_task_TINY.csv",
			output_file_path_chat_level = "./data_cache/raw_output/chat/multi_task_TINY_output_chat_level.csv",
			output_file_path_user_level = "./data_cache/raw_output/user/multi_task_TINY_output_user_level.csv",
			output_file_path_conv_level = "./data_cache/raw_output/conv/multi_task_TINY_output_conversation_level.csv",
			turns = False,
			conversation_id = "stageId",
			cumulative_grouping = False
		)
		feature_builder.featurize(col="message")

		# Set up Dataset Class
		HorseRaceDataSet(data_path = "./data_cache/raw_output/conv/multi_task_TINY_output_conversation_level.csv",
			min_num_chats = 2,
			composition_vars = ["birth_year", "CRT"],
			task_name_mapping = {
			"Sudoku": "Sudoku",
            "Moral Reasoning": "Moral Reasoning (Disciplinary Action Case)",
            "Writing Story": "Writing story",
            "Room Assignment": "Room assignment task",
            "Allocating Resources": "Allocating resources to programs",
            "Divergent Association": "Divergent Association Task",
            "Word Construction": "Word construction from a subset of letters",
        	}
		)
	# Save Dataset Class in Cache
