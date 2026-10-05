import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pylab as plt
import seaborn as sns

# All features Engineering will be done in this script, and kept separated from the general preProcessing

# The first obvious feature to add will be the Molecular Weights (From PubChem)
MWs = {"auraptene": 298.4,
       "osthole": 244.28,
       "esculetin": 178.14,
       "galbanicacid": 398.5,
       "umbelliprenin": 366.5,
       "umbelliferone": 162.14,
       "scopoletin": 192.17,
       "imperatorin": 270.28,
       "herniarin": 176.17};

# The second Important Feature will be LogP (XLogP3 From PubChem)
LogPs = {"auraptene": 5.3,
         "osthole": 3.8,
         "esculetin": 1.2,
         "galbanicacid": 5.1,
         "umbelliprenin": 7.1,
         "umbelliferone": 1.6,
         "scopoletin": 1.5,
         "imperatorin": 3.4,
         "herniarin": 1.9}

# Number of the Hydrogen bond donors
numHDonors = {"auraptene": 0,
              "osthole": 0,
              "esculetin": 2,
              "galbanicacid": 1,
              "umbelliprenin": 1,
              "umbelliferone": 0,
              "scopoletin": 1,
              "imperatorin": 0,
              "herniarin": 0}

# Number of the Hydrohen bond acceptors
numHAcceptors = {"auraptene": 3,
                 "osthole": 3,
                 "esculetin": 4,
                 "galbanicacid": 5,
                 "umbelliprenin": 3,
                 "umbelliferone": 3,
                 "scopoletin": 4,
                 "imperatorin": 4,
                 "herniarin": 3}

# Number of rotatable bonds
rotatableBonds = {"auraptene": 6,
                  "osthole": 3,
                  "esculetin": 0,
                  "galbanicacid": 6,
                  "umbelliprenin": 9,
                  "umbelliferone": 0,
                  "scopoletin": 1,
                  "imperatorin": 3,
                  "herniarin": 1}

# Topological Polar Surface Area (From PubChem)
TPSAs = {"auraptene": 35.5,
         "osthole": 35.5,
         "esculetin": 66.8,
         "galbanicacid": 72.8,
         "umbelliprenin": 35.5,
         "umbelliferone": 46.5,
         "scopoletin": 55.8,
         "imperatorin": 48.7,
         "herniarin": 35.5}

# Ring Counts
ringCounts = {"auraptene": 2,
              "osthole": 2,
              "esculetin": 2,
              "galbanicacid": 3,
              "umbelliprenin": 2,
              "umbelliferone": 2,
              "scopoletin": 2,
              "imperatorin": 2,
              "herniarin": 2}

# FractionCsp3: The fraction of carbon atoms that are sp3 hybridized. (From SwissAdme.ch)
fCsp3s = {"auraptene": 0.32,
          "osthole": 0.27,
          "esculetin": 0,
          "galbanicacid": 0.5,
          "umbelliprenin": 0.38,
          "umbelliferone": 0,
          "scopoletin": 0.10,
          "imperatorin": 0.19,
          "herniarin": 0.10}


# Read the data
data = pd.read_csv("Data/Data.csv")

# Remove null columns
data = data.dropna(axis=1)

# Remove whitspace from the categorical values and make them lowercase
data["Coumarin"] = data["Coumarin"].str.replace(r'\s+', '', regex=True)
data["Coumarin"] = data["Coumarin"].str.lower()

data["Coumarin"] = data["Coumarin"].str.replace(r'\s+', '', regex=True)
data["Coumarin"] = data["Coumarin"].str.lower()

data["Coumarin"] = data["Coumarin"].str.replace(r'\s+', '', regex=True)
data["Coumarin"] = data["Coumarin"].str.lower()

# Remove whitespace from the feature names as well
data.columns = data.columns.str.replace(r'\s+', '', regex=True)

# Add the new features
data["MW"] = data["Coumarin"].map(MWs)
data["LogP"] = data["Coumarin"].map(LogPs)
data["NumHDonors"] = data["Coumarin"].map(numHDonors)
data["NumHAcceptors"] = data["Coumarin"].map(numHAcceptors)
data["RotatableBonds"] = data["Coumarin"].map(rotatableBonds)
data["TPSA"] = data["Coumarin"].map(TPSAs)
data["RingCounts"] = data["Coumarin"].map(ringCounts)
data["fCsp3"] = data["Coumarin"].map(fCsp3s)

# Take a look at the new data structure and columns
print("Head of the data after new features:")
print(tabulate(data.head(), headers='keys', tablefmt='rounded_grid'))

# Check out the correlation matrix of the new features (Ring counts excluded due to low variation)
cols = ["MW", "LogP", "NumHDonors", "NumHAcceptors", "RotatableBonds", "TPSA","RingCounts", "fCsp3"]

corr_matrix = data[cols].corr()
print("\nCorrelation Matrix of the newly engineered features:")
print(tabulate(corr_matrix, headers='keys', tablefmt='rounded_grid'))

# Create heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, 
            annot=True,                # Show values
            cmap='RdBu_r',             # Color scheme
            center=0,                  # Center colormap at 0
            fmt='.2f',                 # 2 decimal places
            square=True,               # Square cells
            linewidths=0.5,            # Grid lines
            cbar_kws={"shrink": 0.8})  # Colorbar size

plt.title('Engineered Features Correlation Matrix Heatmap', fontsize=14)
plt.tight_layout()

# Given the results of the Correlation matrix I have decided that adding cross features could be
# of negative effect to the overall generalization process.

# Save the data
data.to_csv("Data/Engineered_Data.csv",index=False)