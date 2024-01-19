# tpm-horse-race-modeling

The aim of this repository is to organize the data cleaning and model building code for the subproject of #team-process-map that focuses on studying the relative influence of different facets of teamwork. Our key research question is to understand how different aspects of teamwork (the team composition, team size, task features, communication features, etc.) influence a team's eventual performance --- how might we quantify what really "matters" in teams, and build theories towards understanding how the different ingredients of a team work together?

The blueprint for this repository is as follows:

```
├── datasets/ (This is where we need to move the data cleaning.)
├── features/
│   ├── team-process-map/ (This is a git submodule that defines the communication features.)
│   ├── task-mapping/ (This is a git submodule that defines the task features.)
│   ├── team_composition.py (This is where we will clean up team composition features; most of these come in the original data cleaning.)
├── data.py **Class**: defines a Dataset() class that organizes data for modeling in a structured format
├── models/
│   ├── model_instance.py **Class**: defines a Model() object. Starting implementations: LASSO, Ridge, Random Forest ... etc. (all the basics that we had before)
├── experiment_configs/  (.yaml files where we define the kinds of experiments we want to run)
├── data_cache/ (Where we save Dataset() objects as .pkl files)
├── model_cache/ (Where we save Model() objects as .pkl files
├── viz/
│   ├── visualization_tools.py (Saves all the previous tools we had for visualization)
│   ├── viz.ipynb (Sandbox for exploring data viz)
├── sandbox.ipynb (Any other fun explorations to do in a notebook)
├── run_model.py **Main Driver**: This is what we use to actually run everything else; the equivalent in the featurizer is `featurize.py`, which simply calls the other classes.
└── .gitignore
```

The vision is to encapsulate the data cleaning (`data.py`) and modeling (`models`) capacities, so that different iterations of models, feature selection, etc. can be efficiently tested and iterated upon. The repository also aims to ensure clean, reproducible code so that the project can be open-sourced (and replicated!) upon completion.
