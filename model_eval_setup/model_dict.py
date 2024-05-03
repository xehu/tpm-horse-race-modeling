"""
model_dict.py

This is the list of models and the specific hyperparameters that we are searching for in each.
----
NOTE: in order to add support for any other model, we simply need to add a new entry in the model dictionary.
"""
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

model_dict = {"MLPRegressor":{"model": MLPRegressor,
					"params":[{"name":"activation", "type":"choice", "values":["identity", "logistic", "tanh", "relu"]},
							{"name":"solver", "type":"choice", "values":["adam"]}, #"sgd" commented out
                            {"name": "alpha", "type": "range", "bounds":[0.0001,1.0]},
                            {"name":"learning_rate", "type":"choice", "values":["constant", "adaptive"]},
                            {"name":"learning_rate_init", "type":"range", "bounds":[0.0001, 0.01]},
                            {"name":"max_iter", "type":"range", "bounds":[200, 800]}
                        ]},
				"RandomForestRegressor":{"model":RandomForestRegressor,
					"params":[{"name":"n_estimators", "type":"range", "bounds":[2,200]},
							{"name":"max_depth", "type":"range", "bounds":[1,10]},
							{"name":"min_samples_leaf", "type":"range", "bounds":[1,5]}]},
			   "ElasticNet":{ "model": ElasticNet,
					"params":[{"name": "alpha", "type": "range", "bounds":[0.0,2.0]},
							{"name": "l1_ratio", "type": "range", "bounds":[0.0,1.0]},
							{"name": "max_iter", "type": "range", "bounds":[200, 2000]},
							{"name": "selection", "type": "choice", "values":["cyclic", "random"]}
						]},
				"ElasticNetTaskCommunication":{ "model": ElasticNet,
					"params":[{"name": "alpha", "type": "range", "bounds":[0.0001,0.2]}, # previous empirical bounds for task attributes/task complexity/communication: tended to be very small (0.005, 0.17)
							{"name": "l1_ratio", "type": "range", "bounds":[0.0,1.0]},
							{"name": "max_iter", "type": "range", "bounds":[200, 2000]},
							{"name": "selection", "type": "choice", "values":["cyclic", "random"]}
						]},
				"ElasticNetComposition":{ "model": ElasticNet,
					"params":[{"name": "alpha", "type": "range", "bounds":[1.0,2.0]}, # previous empirical bounds that worked better for composition: tended to be between 1 and 2 (1.33, 1.65)
							{"name": "l1_ratio", "type": "range", "bounds":[0.0,1.0]},
							{"name": "max_iter", "type": "range", "bounds":[200, 2000]},
							{"name": "selection", "type": "choice", "values":["cyclic", "random"]}
						]},
				"ElasticNetPlayerCount":{ "model": ElasticNet,
					"params":[{"name": "alpha", "type": "range", "bounds":[0.1,2.1]}, # previous empirical bounds that worked better for team size: tended to vary (0.131, 2.08)
							{"name": "l1_ratio", "type": "range", "bounds":[0.0,1.0]},
							{"name": "max_iter", "type": "range", "bounds":[200, 2000]},
							{"name": "selection", "type": "choice", "values":["cyclic", "random"]}
						]}
			}