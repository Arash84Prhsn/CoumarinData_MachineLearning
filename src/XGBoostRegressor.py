import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from tabulate import tabulate
import os

# Load the data
data = pd.read_csv("Data/Engineered_Data.csv")

# Create some constants
LINE = "="*180

# Check for directories
if not os.path.exists("Results/"):
    os.mkdir("Results/");

# Take a look at the head of the data
print("Head of the raw data:\n", LINE, sep="")
print(tabulate(data.head(), headers='keys'))
print(LINE, "\n")
