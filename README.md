# tpm-horse-race-modeling

The aim of this repository is to organize the data cleaning and model building code for the subproject of #team-process-map that focuses on studying the relative influence of different facets of teamwork. Our key research question is to understand how different aspects of teamwork (the team composition, team size, task features, communication features, etc.) influence a team's eventual performance --- how might we quantify what really "matters" in teams, and build theories towards understanding how the different ingredients of a team work together?

The blueprint for this repository is as follows:

```
├── tpm-data/ (A submodule where data associated with the Team Process Mapping project live.)
│   ├── cleaned_data/ (The folder where datasets that have been cleaned into a format appropriate for team processing mapping reside)
│   ├── data_cleaning/ (Scripts for cleaning the data.)
│   ├── data_cleaning_dev/ (.ipynb files for cleaning the data.)
│   ├── raw_data/ (Uncleaned, unprocessed data.)
│   ├── vector_data/ (A folder where processed, tokenized versions of the datasets reside; e.g., SBERT vectors.) 
├── features/
│   ├── team-process-map/ (This is a git submodule that defines the communication features.)
│   ├── task-mapping/ (This is a git submodule that defines the task features.)
├── data.py **Class**: defines a HorseRaceDataset() class that organizes data for modeling in a structured format
├── model.py **Class**: defines a HorseRaceModel() object. Starting implementations: LASSO, Ridge, Random Forest ... etc. (all the basics that we had before)
├── data_config/  (.yaml files where the user defines requirements for cleaning the data)
├── model_config/  (.yaml files where the user defines requirements for building models)
├── data_cache/ (Where we save Dataset() objects as .pkl files)
├── model_cache/ (Where we save Model() objects as .pkl files)
├── clean_data.py **Model Driver**: The driver for cleaning the data and creating a HorseRaceDataSet instance.
├── run_model.py **Model Driver**: The driver for running and building models, creating a HorseRaceModel instance.
├── viz/
│   ├── visualization_tools.py (Saves all the previous tools we had for visualization)
│   ├── viz.ipynb (Sandbox for exploring data viz)
├── sandbox.ipynb (Any other fun explorations to do in a notebook)
└── .gitignore
```

The vision is to encapsulate the data cleaning (`data.py`) and modeling (`model.py`) capacities, so that different iterations of models, feature selection, etc. can be efficiently tested and iterated upon. The repository also aims to ensure clean, reproducible code so that the project can be open-sourced (and replicated!) upon completion.
