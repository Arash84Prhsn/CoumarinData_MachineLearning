import pandas as pd
import numpy as np
import json
import os

from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, balanced_accuracy_score, recall_score
from tabulate import tabulate

# Path Constants
DATA_DIR = "Data/Processed_Data.csv"
RESULTS_DIR = "Results/Regression/"
FIGURES_DIR = RESULTS_DIR + "Figures/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"

# mkdirs commands to ensure that the directories do exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)

# Training constants
RANDOM_STATE = 42
N_ITER = 30 # For the randomsearch to find the best parameters
N_SPLITS = 5 # For KFold

# Other constants
LINE = "=" * 100
TRIANGLE = "▶"