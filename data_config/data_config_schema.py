# data_config_schema.py
data_config_schema = {
    "data_paths": {
        "raw_round_data_path": str,
        "raw_stage_data_path": str,
        "raw_user_data_path": str,
        "vector_directory": str,
        "output_cleaned": str,
        "output_chat": str,
        "output_user": str,
        "output_conv": str,
        "output_pkl": str
    },
    "conv_featurizer_options":{
        "analyze_first_pct":[1.0], 
        "turns": True,
        "conversation_id": None,
        "cumulative_grouping": False, 
        "within_task": False
    },
    "cleaning_options": {
        "min_num_chats": int,
        "num_conversation_components": None,
        "handle_na_values": "mean",
        "na_fill_value": None, 
        "standardize_dv": True,
        "use_mean_for_roundId": False,
        "tiny": False
    },
    "format_options":{
        "task_name_index": 0,
        "complexity_name_index": 1,
        "total_messages_varname": "sum_num_messages",
        "team_size_varname": "playerCount"
    },
    "variable_options":{
        "dvs": None,
        "composition_vars": None,
        "task_vars": None,
        "task_name_mapping": None
    }
}