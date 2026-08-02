import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from tabulate import tabulate
import os


# Load the data
data = pd.read_csv()