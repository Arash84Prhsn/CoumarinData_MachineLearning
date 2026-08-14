import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from tabulate import tabulate
import os


# Load the data
data = pd.read_csv("Data/Final_Data.csv")

LINE = "="*166

# Take a look at the head of the data
print(tabulate(data.head(), headers='keys', tablefmt='grid'))
print(LINE)

# Add the new classification feature
data["ViabilityClass"] = (data['Viability'].sub(50).abs() <= 5).astype(int)

# Check out the distribution of the classes
print("Class Distribution:")
print(data['ViabilityClass'].value_counts())
print(LINE)

# Head of the data after adding the new feature:
print(tabulate(data.head(), headers='keys', tablefmt='grid'))
print(LINE)
