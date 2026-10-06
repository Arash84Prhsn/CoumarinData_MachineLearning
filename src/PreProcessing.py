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
data1 = pd.read_csv("Data/Data.csv")
data2 = pd.read_csv("Data/new_data.csv")

# Fix time in data2
data2["Time"] = data2["Time"].str.replace("h", "", regex=False).astype(float)

# Remove the null columns and rows from the datasets
data2 = data2 [["Author", "Cancer", "CellLine", "Coumarin", "Dose", "Time", "Viability"]]
data2 = data2.dropna(axis=0)
data1 = data1.dropna(axis=1)

# Concatenate the datasets:
data = pd.concat(objs=[data1, data2])

# Print the sizes of the datasets:
print()
print("Size of the first dataset: ", data1.__len__())
print("Size of the second dataset: ", data2.__len__())
print("Size of the combined dataset: ", data.__len__())

line = "=" * 100 + "\n"
print(line)

# Coumarins in each dataset:
print('Data1 coumarins:')
print(data1["Coumarin"].value_counts())
print()
print('Data2 coumarins:')
print(data2["Coumarin"].value_counts())
print(line)


# Remove whitspace from the categorical values and make them lowercase
data["Coumarin"] = data["Coumarin"].str.replace(r'\s+', '', regex=True)
data["Coumarin"] = data["Coumarin"].str.lower()

data["Cancer"] = data["Cancer"].str.replace(r'\s+', '', regex=True)
data["Cancer"] = data["Cancer"].str.lower()

data["CellLine"] = data["CellLine"].str.replace(r'\s+', '', regex=True)
data["CellLine"] = data["CellLine"].str.lower()

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
print(data.head())
print(line)

# Check out the correlation matrix of the new features (Ring counts excluded due to low variation)
cols = ["MW", "LogP", "NumHDonors", "NumHAcceptors", "RotatableBonds", "TPSA","RingCounts", "fCsp3"]

# Print all of our features:
print()
print("Features of our data:")
print(data.columns.to_list())

corr_matrix = data[cols].corr()
print("\nCorrelation Matrix of the newly engineered features:")
print(tabulate(corr_matrix, headers='keys', tablefmt='rounded_grid'))

# Create heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, 
            annot=True,                # Show values
            cmap='Spectral',             # Color scheme
            center=0,                  # Center colormap at 0
            fmt='.2f',                 # 2 decimal places
            square=True,               # Square cells
            linewidths=0.5,            # Grid lines
            cbar_kws={"shrink": 0.8})  # Colorbar size

plt.title('Structural Features Correlation Matrix', fontsize=20)
plt.tight_layout()
plt.savefig("Figures/Correlation_Matrix.png")
# plt.show()

# Save the data
data.to_csv("Data/Processed_Data.csv",index=False)

# Make data for classification models as well
# Class 1 is when viability is between 40 and 60, class zero is otherwise
classification_data = data.copy()

classification_data["Viability"] = (
    (classification_data["Viability"] > 40) &
    (classification_data["Viability"] < 60)
).astype(int)

print()
print(line)
print("Classification target distribution:")
print(classification_data["Viability"].value_counts().sort_index())

print()
print("Classification target proportions:")
print(classification_data["Viability"].value_counts(normalize=True).sort_index())

classification_data.to_csv(
    "Data/Classification_Data.csv",
    index=False
)

print()
print("Classification data saved to: Data/Classification_Data.csv")