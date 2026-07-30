import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate

# Load the data
data = pd.read_csv("Data/Engineered_Data.csv")

# Set some constants
LINE = "="*180
RESULTS_DIR = "Results/"

# Take a look at the head of the data
print("Head of the raw data:\n", LINE, sep="")
print(tabulate(data.head(), headers='keys'))
print(LINE, "\n")

