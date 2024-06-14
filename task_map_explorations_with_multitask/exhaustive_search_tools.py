import pandas as pd
import random
import ast
import numpy as np
import scipy.stats as stats 
from scipy.spatial import distance
import statsmodels.stats.multitest as multitest

import statsmodels.stats.api as sms
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
import os, glob
import sys
import pickle
import itertools
import concurrent.futures
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import csv
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tqdm import tqdm

plt.rcParams["font.family"] = "Times New Roman"

import warnings
warnings.filterwarnings("ignore")

full_multi_task_data = pd.read_csv('all_multi_task_wave_data.txt')
full_multi_task_data = full_multi_task_data.rename(columns = {"task": "task_name"})
# set the score to the best score across repeated attempts, in cases where it saved multiple times
full_multi_task_data = full_multi_task_data.groupby('stageId').apply(lambda x: x.loc[x['score'].idxmax()]).reset_index(drop=True)

# task map
task_map = pd.read_csv('task_map.csv')
task_map = task_map.rename(columns = {"task": "task_name"})

task_name_mapping = {
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

task_map = task_map.rename(
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

full_multi_task_data.loc[:, "task_name"] = full_multi_task_data["task_name"].replace(task_name_mapping)
task_cols_to_use = task_map.drop(["task_name", "Type 6 (Mixed-Motive)"], axis = 1).columns
# merge the multi-task data with the task map
full_multi_task_data = pd.merge(left = full_multi_task_data, right = task_map, on = "task_name", how = "left")
communication_features = pd.read_csv("full_multi_task_messages_conversation_level.csv")
communication_features = communication_features.rename(columns={"conversation_num": "stageId"})
communication_features.columns
communication_features = communication_features.drop(columns = ['speaker_nickname', 'message',
       'timestamp', 'message_original', 'message_lower_with_punc'], axis = 1)
COMMS_DVS = ["turn_taking_index", "gini_coefficient_sum_num_messages", "sum_num_messages", "average_positive_bert", "team_burstiness"]

# Final Cleaned Datasets
team_multi_task_full = full_multi_task_data[full_multi_task_data["playerCount"]>1]
team_multi_task_wave1 = team_multi_task_full[team_multi_task_full["wave"]==1]
team_multi_task_comms_full = pd.merge(communication_features, team_multi_task_full, on = "stageId", how = "inner")
team_multi_task_comms_wave1 = pd.merge(communication_features, team_multi_task_wave1, on = "stageId", how = "inner")

# Design decision: Drop all cases where the communication features are missing
team_multi_task_comms_full.dropna(subset = COMMS_DVS, inplace = True)
team_multi_task_comms_wave1.dropna(subset = COMMS_DVS, inplace = True)

cols_to_use = ["score"] + list(task_cols_to_use)
cols_to_use_with_comms = COMMS_DVS + ["score"] + list(task_cols_to_use)

team_multi_task_full[cols_to_use] = StandardScaler().fit_transform(team_multi_task_full[cols_to_use])
team_multi_task_wave1[cols_to_use] = StandardScaler().fit_transform(team_multi_task_wave1[cols_to_use])
team_multi_task_comms_full[cols_to_use_with_comms] = StandardScaler().fit_transform(team_multi_task_comms_full[cols_to_use_with_comms])
team_multi_task_comms_wave1[cols_to_use_with_comms] = StandardScaler().fit_transform(team_multi_task_comms_wave1[cols_to_use_with_comms])

# Q^2
def reshape_x_y(x, y):
    if(isinstance(x, pd.Series)):
        x = np.asarray(x).reshape(-1, 1)
    else:
        x = np.asarray(x)
    
    y = np.asarray(y).reshape(-1, 1)
    return(x, y)

def q2_task_holdout_helper(x_train, x_test, y_train, y_test, estimator):
    
    # some reshaping
    x_train_array, y_train_array = reshape_x_y(x_train, y_train)
    x_test_array, y_test_array = reshape_x_y(x_test, y_test)

    # print("Training data: ", pd.DataFrame(x_train_array).head())
    # print("Testing data: ", pd.DataFrame(x_test_array).head())

    # Fit the model and get the error
    fitted_model = estimator.fit(X=x_train_array, y=y_train_array.ravel())
    
    # save prediction error
    prediction = fitted_model.predict(x_test_array)

    # flatten all arrays
    y_test_array = np.asarray(y_test_array).flatten()
    prediction = np.asarray(prediction).flatten()

    # print("y test array", y_test_array)
    # print("prediction", prediction)

    squared_model_prediction_error = (y_test_array - prediction) ** 2

    # save total error for this fold
    squared_average_prediction_error = (y_test_array - np.mean(y_train_array)) ** 2

    return squared_model_prediction_error, squared_average_prediction_error

"""
This is the version of q^2 that holds out EVERYTHING associated with a given task

It trains on all task instances from the "seen" classes, and it tests on task instances of held-out (unseen) classes.

NOTE: this version of the function assumes that x and y are passed in with a column called "task_name"
"""

def get_q2(y, x, estimator = Lasso(), num_task_holdouts = 1):

    squared_model_prediction_errors = []
    squared_average_prediction_errors = []

    num_total_tasks = x["task_name"].nunique()

    # randomly hold out `num_task_holdouts`
    all_possible_task_combos = list(itertools.combinations((x["task_name"].unique()), num_total_tasks - num_task_holdouts))
    
    for sample in all_possible_task_combos:

        # print("Sample:", sample)
        # print("Held out:", x[~x["task_name"].isin(sample)]["task_name"].unique())

        x_train_tasks = x[x["task_name"].isin(sample)].drop("task_name", axis = 1)
        x_test_tasks = x[~x["task_name"].isin(sample)].drop("task_name", axis = 1)

        y_train_tasks = y[y["task_name"].isin(sample)].drop("task_name", axis = 1)
        y_test_tasks = y[~y["task_name"].isin(sample)].drop("task_name", axis = 1)

        # get evaluation score by training on the training tasks and evaluating on the holdout tasks
        squared_model_prediction_error, squared_average_prediction_error = q2_task_holdout_helper(x_train_tasks, x_test_tasks, y_train_tasks, y_test_tasks, estimator)
        
        squared_model_prediction_errors.append(squared_model_prediction_error)
        squared_average_prediction_errors.append(squared_average_prediction_error)

    squared_model_prediction_error = np.asarray(squared_model_prediction_error).flatten()
    squared_average_prediction_error = np.asarray(squared_average_prediction_error).flatten()

    return 1 - (np.sum(squared_model_prediction_error) / np.sum(squared_average_prediction_error))

### EXHAUSTIVE SEARCH PROCEDURE ###

def process_combination(task_col_combo, dataset, dv, filename):
    return get_q2(
                dataset[[dv, "task_name"]],
                dataset[list(task_col_combo) + ["playerCount", "Low", "Medium", "task_name"]],
                estimator = LinearRegression() ## get_q2 defaults to LASSO, so let's run this with OLS
    )

"""
function: parallel_q2
---
This is the function that (in parallel) gets the q^2 values of each
of the possible combinations / ways of selecting task columns.

@dataset: the dataset that we are using for the prediction
@dv: the name of the dependent variable. We expect this to be a column in the dataset.
@filename: the name of the output file (as we finish processing the combinations, the results will be written to this file).
@column_choice_combinations: the list of all possible column choice combinations.
"""
def parallel_q2(dataset, dv, filename, column_choice_combinations, results_df):
    num_threads = multiprocessing.cpu_count()  # Get as many processes as CPU's
    existing_set = set(results_df["selected_task_cols"])

    # append if it exists, and write a new file if it doesn't yet exist
    if (not os.path.isfile(filename)):
        results_df.to_csv(filename, index = False)
        print("making new CSV...")

    csv_opened = open(filename, 'a', newline='')

    with csv_opened as csvfile:
        fieldnames = ['selected_task_cols', 'q2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        queue = Queue()

        # This is the thread that consumes and appends to the CSV
        def consume():
            while True:
                if not queue.empty():
                    completed_tuple = queue.get()
                    # Row comes out of queue; CSV writing goes here
                    writer.writerow({'selected_task_cols': completed_tuple[0], 'q2': completed_tuple[1]})

        consumer = threading.Thread(target=consume)
        consumer.setDaemon(True)
        consumer.start()

        with tqdm(total=len(column_choice_combinations)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

                # all these threads calculate q2 and put them into the queue
                def produce(task_col_combo):
                    # Data processing goes here; row goes into queue
                    if str(task_col_combo) not in existing_set:
                        q2 = process_combination(task_col_combo, dataset, dv, filename)                    
                        result_to_append = (str(task_col_combo), q2)
                        queue.put(result_to_append)

                futures = [executor.submit(produce, combo) for combo in column_choice_combinations]

                for future in concurrent.futures.as_completed(futures):
                    pbar.update()

            consumer.join()

"""
Sometimes the CSV will have a corrupted last line, from having run only partway through when the kernel died. 
We need to delete the final line in order to prevent an EOF error.
We subsequently add a new line at the end for easy appending.
"""
def delete_last_line(filename):
    with open(filename, "r+", encoding = "utf-8") as file:
        # Move the pointer (similar to a cursor in a text editor) to the end of the file
        file.seek(0, os.SEEK_END)

        # This code means the following code skips the very last character in the file -
        # i.e. in the case the last line is null we delete the last line
        # and the penultimate one
        pos = file.tell() - 1

        # Read each character in the file one at a time from the penultimate
        # character going backwards, searching for a newline character
        # If we find a new line, exit the search
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)

        # So long as we're not at the start of the file, delete all the characters ahead
        # of this position
        if pos > 0:
            file.seek(pos, os.SEEK_SET)
            file.truncate()

    # add a new line at the end
    with open(filename,'a') as f:
        f.write("\n")

def exhaustive_search_for_ncol_and_dv(ncol, dv, is_full):
    print("Running exhaustive search for " + str(ncol) + " columns...")
    column_choice_combinations = list(itertools.combinations(task_cols_to_use, ncol))
    
    lock = threading.Lock()
    
    name = "q2_OLS_from_diff_task_cols"
    if is_full:
        name = name + "_FULL"
    name = name + ".csv"
    
    output_filename = str(ncol) + "_" + dv + "_" + name
    
    # read in the results from an existing output, if we have it
    if os.path.isfile(output_filename):
        print(output_filename)
        delete_last_line(output_filename)
        results_df = pd.read_csv(output_filename)
        print("length of current results dataframe")
        print(len(results_df))
        results = list(results_df.itertuples(index=False, name=None))
    else:
        print("no current results dataframe found, starting anew...")
        results = []
        results_df = pd.DataFrame({"selected_task_cols":[], "q2": []})
    
    if dv == "score":
        # this is the dataset without conversational features
        if is_full:
            data = team_multi_task_full
        else:
            data = team_multi_task_wave1
    else:
        # this is the dataset WITH conversational features
        if is_full:
            data = team_multi_task_comms_full
        else:
            data = team_multi_task_comms_wave1


    parallel_q2(dataset = data , dv = dv, filename = output_filename,
            column_choice_combinations = column_choice_combinations, results_df = results_df)