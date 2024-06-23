"""
file: data.py
-------------
This file defines the Dataset class that organizes and cleans the dataset
into a structured file format.

The objectives of this class are:
- Read in the data source
- Break the data source into the appropriate collections of features
	{Task, Communication, Composition, Team Size ...} and DV's
- Apply any hand-chosen filters (e.g., looking at only a subset of the columns)
- Store these features/DV's into attributes of the class

Key attributes of an instance of this class:
- self.task_name: the column containing task names
- self.task_features: the Task Features
- self.task_complexity_features: the Task Complexity Features
- self.composition_features: the Team Composition Features
- self.size_feature: the Team Size Feature
- self.conversation_features: the Team Conversation Features (from TPM)
- self.dvs: the dependent variables

"""

# 3rd Party Imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class HorseRaceDataSet:
    def __init__(
    	self,
    	data_path: str,
        min_num_chats: int,
        num_conversation_components = None,
        task_name_index = 0,
        total_messages_varname = "sum_num_messages",
        team_size_varname = "playerCount",
        handle_na_values = "mean",
        custom_task_cols = [],
        na_fill_value = None,
        standardize_dv = True,
        standardize_iv = True,
        dvs = None,
        composition_vars = None,
        task_vars = None,
        custom_task_predictors = None,
        task_name_mapping = None,
        complexity_to_drop = None
    ) -> None:
        """
        Parameters:
        
        1. Required Parameters:
        @data_path: the source of the main data output (after it has been passed through the featurizer)
        @min_num_chats: the minimum number of chats that teams need to have had in order to be included in the analysis.
        
        2. Optional Parameters
        @num_conversation_components (Defaults to None): if an int is passed in, this is the number of PC's to use when analyzing conversation data.
        @task_name_index (Defaults to 0): in the parameter `task_vars`, the index of the item that contains the task name.
        @complexity_name_index (Defaults to 1): in the parameter `task_vars,` the index of the item that contains the complexity
        @total_messages_varname (Defaults to "sum_num_messages"): The name of the variable that tracks the number of messages per conversation
        @team_size_varname (Defaults to "playerCount"): The name of the variable that tracks the number of members/players per team
        @handle_na_values (Defaults to "mean"): determines how to handle missing values; accepted values are {"drop", "mean", "median", "most_frequent", or "constant"}. 
            If "constant," must also provide a fill value (`na_fill_value`)
        @na_fill_value: the value with which the user specifies filling NA values.
        @standardize_dv (defaults to True): standardize the dependent variable(s)
        @standardize_iv (defaults to True): standardize the independent variable(s)
            IV's that are candidates for being standardized are:
                - Conversation Features
                - Task Features
                - Composition Features
            IV's that are not currently candidates for being standardized are:
                - Any binary variables
                - Team size (there are only 2 options, 3 and 6, in the multi-task data)
                - Task Complexity (this is one-hot encoded)
        If the following optional parameters are None, they are set to values specific to the multi-task dataset, defined later in the constructor.
        @dvs (Defaults to None): custom list of dependent variables of interest
        @composition_vars (Defaults to None): custom list of composition variables of interest
        @task_vars (Defaults to None): custom list of task variable names in the data (this is the *target dataset*, not the Task Map!)
        @custom_task_predictors (Defaults to None): custom list of task predictors (e.g., columns in the Task Map) desired by the user; otherwise, we use the full space!
        @task_name_mapping (Defaults to None): custom dictionary mapping variable names in the dataset (keys) to Task Map variable names (values)
        @complexity_to_drop (Defaults to None): drops a complexity column (e.g., "Low", "Medium", "High") as the reference class
        """

        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)

        task_map_path = './features/task-mapping/task_map.csv'
        self.task_map = pd.read_csv(task_map_path)
        self.task_map = self.task_map.rename(
            columns = {
                "Q1concept_behav": "Conceptual-Behavioral",
                "Q3type_1_planning": "Type 1 (Planning)",
                "Q4type_2_generate": "Type 2 (Generate)",
                "Q6type_5_cc": "Type 5 (Cognitive Conflict)",
                "Q7type_7_battle": "Type 7 (Battle)",
                "Q8type_8_performance": "Type 8 (Performance)",
                "Q9divisible_unitary": "Divisible-Unitary",
                "Q10maximizing": "Maximizing",
                "Q11optimizing": "Optimizing",
                "Q13outcome_multip": "Outcome Multiplicity",
                "Q14sol_scheme_mul": "Solution Scheme Multiplicity",
                "Q15dec_verifiability": "Decision Verifiability",
                "Q16shared_knowledge": "Shared Knowledge",
                "Q17within_sys_sol": "Within-System Solution",
                "Q18ans_recog": "Answer Recognizability",
                "Q19time_solvability": "Time Solvability",
                "Q20type_3_type_4": "Type 3 and Type 4 (Objective Correctness)",
                "Q22confl_tradeoffs": "Conflicting Tradeoffs",
                "Q23ss_out_uncert": "Solution Scheme Outcome Uncertainty",
                "Q24eureka_question": "Eureka Question",
                "Q2intel_manip_1" : "Intellectual-Manipulative",
                "Q21intellective_judg_1" : "Intellective-Judgmental",
                "Q5creativity_input_1" : "Creativity Input",
                "Q25_type6_mixed_motive" : "Type 6 (Mixed-Motive)"
            }
        )

        self.min_num_chats = min_num_chats
        self.num_conversation_components = num_conversation_components
        self.task_name_index = task_name_index
        self.total_messages_varname = total_messages_varname
        self.team_size_varname = team_size_varname

        # Assert that the na handling options are valid
        assert handle_na_values in {"drop", "mean", "median", "most_frequent", "constant"}, "Invalid specification of handling NA values."
        if handle_na_values == "drop":
            self.drop_na = True
            self.imputation_option = None
        else:
            self.drop_na = False
            self.imputation_option = handle_na_values

        self.na_fill_value = na_fill_value
        self.standardize_dv = standardize_dv
        self.standardize_iv = standardize_iv
        self.complexity_to_drop = complexity_to_drop

        # Default DV's, Composition Variables, and Task Name Map (this is for the multi-task dataset)
        dvs_default = ["score","duration","efficiency"]
        composition_default = ['birth_year', 'CRT', 'income_max', 'income_min', 'IRCS_GS', 'IRCS_GV', 'IRCS_IB', 'IRCS_IR',
                    'IRCS_IV', 'IRCS_RS', 'political_fiscal', 'political_social', 'RME', 'country', 'education_level', 'gender', 'marital_status', 'political_party', 'race', 'playerCount']
        task_vars_default =  ['task', 'Low', 'Medium', 'High']
        task_name_mapping_default = {
            "Sudoku": "Sudoku",
            "Moral Reasoning": "Moral Reasoning (Disciplinary Action Case)",
            "Wolf Goat Cabbage": "Wolf, goat and cabbage transfer",
            "Guess the Correlation": "Guessing the correlation",
            "Writing Story": "Writing story",
            "Room Assignment": "Room assignment task",
            "Allocating Resources": "Allocating resources to programs",
            "Divergent Association": "Divergent Association Task",
            "Word Construction": "Word construction from a subset of letters",
            "Whac a Mole": "Whac-A-Mole",
            "Random Dot Motion": "Random dot motion",
            "Recall Association": "Recall association",
            "Recall Word Lists": "Recall word lists",
            "Typing": "Typing game",
            "Unscramble Words": "Unscramble words (anagrams)",
            "WildCam": "Wildcam Gorongosa (Zooniverse)",
            "Advertisement Writing": "Advertisement writing",
            "Putting Food Into Categories": "Putting food into categories"
        }

        self.dvs = dvs if dvs is not None else dvs_default
        self.composition_vars = composition_vars if composition_vars is not None else composition_default
        # Break out the team size from the composition variables
        self.composition_vars.remove(self.team_size_varname)
        self.task_vars = task_vars if task_vars is not None else task_vars_default
        self.task_name_mapping = task_name_mapping if task_name_mapping is not None else task_name_mapping_default
        
        # Assertions that parameters are the right type
        assert type(self.dvs)==list, "The optional parameter `dvs` should be a list."
        assert type(self.composition_vars)==list, "The optional parameter `composition_vars` should be a list."
        assert type(self.task_vars)==list, "The optional parameter `task_vars` should be a list."
        assert type(self.task_name_mapping)==dict, "The optional parameter `task_name_mapping` should be a dict."

        # set the column containing the task name
        self.task_name_col = self.task_vars[self.task_name_index]

        # set task complexity columns and drop a complexity level if required
        self.complexity_cols = self.task_vars.copy()
        self.complexity_cols.remove(self.task_name_col)
        if (self.complexity_to_drop in self.complexity_cols):
            self.complexity_cols.remove(self.complexity_to_drop)

        # these are the task predictors we will actually use; by default, we include all task columns
        self.custom_task_predictors = custom_task_predictors if custom_task_predictors is not None else self.task_name_mapping.keys()
        assert type(self.custom_task_predictors)==list, "The optional parameter `custom_task_predictors` should be a list."

        # remove the "message" column if present
        if "message" in self.data.columns:
            self.data.drop(["message"], axis=1, inplace=True)

        # handle na's according to user specifications
        self.handle_na_values()

        # preprocess data
        self.read_and_preprocess_data()

    def drop_invariant_columns(self, df) -> pd.DataFrame:
        """
        Certain features are invariant throughout the training data (e.g., the entire column is 0 throughout).

        These feature obviously won't be very useful predictors, so we drop them.
        
        This function works by identifying columns that only have 1 unique value throughout the entire column,
        and then dropping them.

        @df: the dataframe containing the features (this should be X).
        """
        nunique = df.nunique()
        cols_to_drop = nunique[nunique == 1].index
        return(df.drop(cols_to_drop, axis=1))

    def process_task_data(self, data) -> pd.DataFrame:
        task = data[self.task_vars].copy()

        # Merge with Task Map
        task.loc[:, self.task_name_col] = task[self.task_name_col].replace(self.task_name_mapping)
        task = pd.merge(left=task, right=self.task_map, on = self.task_name_col, how='left')

        # filter for the custom task features
        # note that the complexity is *included* in the task features
        task = task[self.custom_task_predictors + self.complexity_cols + [self.task_name_col]]
        
        # Save Task Name in another column
        self.task_name = task[self.task_name_col]

        # Drop task name from the task features
        task.drop([self.task_name_col], axis=1, inplace=True)

        return(task)

    def process_task_complexity(self, data) -> pd.DataFrame:
        return data[self.complexity_cols] # It's already in the data!

    def process_composition_data(self, data) -> pd.DataFrame:
        composition = data[[col for col in data.columns if any(keyword in col for keyword in self.composition_vars)]]
        return(composition)

    def process_team_size_data(self, data) -> pd.DataFrame:
        return data[self.team_size_varname]

    def process_communication_data(self, data) -> pd.DataFrame:  
        # assumes all features that are not dv's or compositions variables are communication variables
        conversation = data.drop(columns= self.dvs+list(self.composition_features.columns))._get_numeric_data()
        return(conversation)

    def standardize_df_by_group(self, df, grouper) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=np.number)
        non_binary_cols = numeric_cols.columns[df.nunique() > 2]  # Identify non-binary columns; binary columns don't standardize well.
        if grouper is not None:
            grouped = df.groupby(grouper)
            df[non_binary_cols] = grouped[non_binary_cols].transform(lambda x: (x - x.mean()) / x.std())
        else:
            df[non_binary_cols] = df[non_binary_cols].transform(lambda x: (x - x.mean()) / x.std())

        return df

    def process_dv_data(self, data) ->  pd.DataFrame:
        dvs = data[self.dvs + [self.task_name_col]]
        # standardize_dv by the task
        if(self.standardize_dv):
            dvs = self.standardize_df_by_group(dvs, grouper = self.task_name_col)

        return(dvs)

    def get_first_n_pcs(self, data, num_components) -> pd.DataFrame:
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(data.transform(lambda x: (x - x.mean()) / x.std()))
        print("PCA explained variance:")
        print(np.sum(pca.explained_variance_ratio_))
        return (pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]))

    """
    Handle NA values according to the user's specifications.
    """
    def handle_na_values(self) -> None:
        numeric_columns = self.data.select_dtypes(include='number').columns
        non_numeric_columns = self.data.columns.difference(numeric_columns)

        # Handle NA values for numeric columns
        numeric_data = self.data[numeric_columns].to_numpy()
        if self.drop_na:
            numeric_data = numeric_data.dropna()
        else:  # apply an imputation option
            if self.imputation_option != "constant":
                imputer = SimpleImputer(strategy=self.imputation_option)
                numeric_data = imputer.fit_transform(numeric_data)
            else:
                assert self.na_fill_value is not None, "You cannot specify a 'constant' NA fill option without specifying an `na_fill_value.`"
                imputer = SimpleImputer(strategy=self.imputation_option, fill_value=self.na_fill_value)
                numeric_data = imputer.fit_transform(numeric_data)

        # Merge numeric and non-numeric data back together
        self.data = pd.concat([pd.DataFrame(numeric_data, columns=numeric_columns), self.data[non_numeric_columns]], axis=1)

    def read_and_preprocess_data(self) -> None:
        """
        This function processes the data from self.data into three sets
        """
        assert set(self.dvs)<= set(list(self.data.columns)), "The desired DV's are not found in the data."
        composition_cols = [] # Composition columns are summarized by mean, std
        for varname in self.composition_vars:
            composition_cols.append(varname + "_mean")
            composition_cols.append(varname + "_std")
        assert set(composition_cols) <= set(self.data.columns), "The following desired composition variables are not found in the data: " + str(set(composition_cols).difference(set(self.data.columns)))
        assert set(self.task_vars)<=set(self.data.columns), "The following desired task variables are not found in the data: " + str(set(self.task_vars).difference(set(self.data.columns)))
        assert set(self.custom_task_predictors)<=set(self.task_map.columns), "The following desired task variables are not found in the Task Map: " + str(set(self.custom_task_predictors).difference(set(self.task_map.columns)))
        assert set(self.task_name_mapping.keys()) <= set(self.data[self.task_name_col].drop_duplicates()), "The desired task names in the dictionary are not found in the data."
        assert self.total_messages_varname in self.data.columns, "There is no variable for the total number of messages in the dataset; this variable is required for processing the data."
        assert self.team_size_varname in self.data.columns, "There is no variable for the total number of members in the team; this variable is required for processing the data."

        print("Passed assertions, cleaning features now ...")

        data = self.data

        # drop invariant columns in data
        data = self.drop_invariant_columns(data)

        # Filter this down to teams that have at least min_num of chats
        data = data[data[self.total_messages_varname] >= self.min_num_chats]

        # Task data
        self.task_features = self.process_task_data(data)
        if(self.standardize_iv):
            self.task_features = self.standardize_df_by_group(self.task_features, grouper = None)

        print("Completed Task Features...")

        # Task complexity data
        self.task_complexity_features = self.process_task_complexity(data)

        print("Completed Task Complexity Features...")

        # Composition data
        self.composition_features = self.process_composition_data(data)
        if(self.standardize_iv):
            self.composition_features = self.standardize_df_by_group(self.composition_features, grouper = None)
         
        print("Completed Composition Features...")

        # Team Size data
        self.size_feature = self.process_team_size_data(data)

        print("Completed Team Size Features...")

        # Conversation data
        self.conversation_features = self.process_communication_data(data)
        if(self.num_conversation_components is not None):
            self.conversation_features = self.get_first_n_pcs(self.conversation_features, self.num_conversation_components)
        if(self.standardize_iv):
            self.conversation_features = self.standardize_df_by_group(self.conversation_features, grouper = None)

        print("Completed Conversation Features...")

        # DVs
        self.dvs = self.process_dv_data(data)
        print("Completed DV's .... All done!")