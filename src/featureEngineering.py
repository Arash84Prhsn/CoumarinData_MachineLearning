import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pylab as plt
import seaborn as sns

# All features Engineering will be done in this script, and kept separated from the general preProcessing

# The first obvious feature to add will be the Molecular Weights (From PubChem)
MWs = {"Auraptene": 298.4,
       "Osthole": 244.28,
       "Esculetin": 178.14,
       "GalbanicAcid": 398.5,
       "Umbelliprenin": 366.5};

# The second Important Feature will be LogP (XLogP3 From PubChem)
LogPs = {"Auraptene": 5.3,
         "Osthole": 3.8,
         "Esculetin": 1.2,
         "GalbanicAcid": 5.1,
         "Umbelliprenin": 7.1}

# Number of the Hydrogen bond donors
numHDonors = {"Auraptene": 0,
              "Osthole": 0,
              "Esculetin": 2,
              "GalbanicAcid": 1,
              "Umbelliprenin": 0}

# Number of the Hydrohen bond acceptors
numHAcceptors = {"Auraptene": 3,
                 "Osthole": 3,
                 "Esculetin": 4,
                 "GalbanicAcid": 5,
                 "Umbelliprenin": 3}

# Number of rotatable bonds
rotatableBonds = {"Auraptene": 6,
                  "Osthole": 3,
                  "Esculetin": 0,
                  "GalbanicAcid": 6,
                  "Umbelliprenin": 9}

# Topological Polar Surface Area (From PubChem)
TPSAs = {"Auraptene": 35.5,
         "Osthole": 35.5,
         "Esculetin": 66.8,
         "GalbanicAcid": 72.8,
         "Umbelliprenin": 35.5}

# Ring Counts
ringCounts = {"Auraptene": 2,
              "Osthole": 2,
              "Esculetin": 2,
              "GalbanicAcid": 3,
              "Umbelliprenin": 2}

# FractionCsp3: The fraction of carbon atoms that are sp3 hybridized.
fCsp3s = {"Auraptene": 0.32,
          "Osthole": 0.27,
          "Esculetin": 0,
          "GalbanicAcid": 0.5,
          "Umbelliprenin": 0.38}


# Read the data
data = pd.read_csv("Data/Final_Data.csv")

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
print(tabulate(data.head(), headers='keys', tablefmt='grid'))

# Check out the correlation matrix of the new features
cols = ["MW", "LogP", "NumHDonors", "NumHAcceptors", "RotatableBonds", "TPSA", "RingCounts", "fCsp3"]
corr_matrix = data[cols].corr()

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

plt.show()