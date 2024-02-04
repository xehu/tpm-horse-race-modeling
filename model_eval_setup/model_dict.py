"""
model_dict.py

This is the list of models and the specific hyperparameters that we are searching for in each.
----
NOTE: in order to add support for any other model, we simply need to add a new entry in the model dictionary.
"""
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

model_dict = {"RandomForestRegressor":{"model":RandomForestRegressor,
					"params":[{"name":"n_estimators", "type":"range", "bounds":[2,256]},
							{"name":"max_depth", "type":"range", "bounds":[1,10]},
							{"name":"min_samples_leaf", "type":"range", "bounds":[1,10]}]},
			   "ElasticNet":{ "model": ElasticNet,
					"params":[{"name": "alpha", "type": "range", "bounds":[0.0,1.0]},
							{"name": "l1_ratio", "type": "range", "bounds":[0.0,1.0]},
							{"name": "max_iter", "type": "range", "bounds":[200, 2000]},
							{"name": "selection", "type": "choice", "values":["cyclic", "random"]}
						]}
			}