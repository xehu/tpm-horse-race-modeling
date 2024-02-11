# model_eval_config_schema.py
model_eval_config_schema = {
    "model_paths": {
        "HorseRaceDataSet_path": str,
        "optimization_output_path": str,
        "evaluation_output_path": str,
    },
    "model_options":{
        "X_cat_names": list, 
        "y_name": str,
        "model_type": str,
        "inversely_weight_tasks": True,
        "handle_na_values": "mean",
        "na_fill_value": None
    },
    "optimization_options": {
        "n_iterations": int,
        "total_trials": int,
        "output_file_mode": "a"
    }
}