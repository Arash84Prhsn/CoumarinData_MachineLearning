import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import seaborn as sns
from tabulate import tabulate
import matplotlib.pylab as plt
import unicodedata
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Compound smiles from PubChem:
SMILES = {
    "auraptene": r"CC(=CCC/C(=C/COC1=CC2=C(C=C1)C=CC(=O)O2)/C)C",
    "osthole": r"CC(=CCC1=C(C=CC2=C1OC(=O)C=C2)OC)C",
    "esculetin": r"C1=CC(=O)OC2=CC(=C(C=C21)O)O",
    "galbanicacid": r"C[C@H]1CCC(=C(C)C)[C@@H]([C@@]1(C)COC2=CC3=C(C=C2)C=CC(=O)O3)CCC(=O)O",
    "umbelliprenin": r"CC(=CCC/C(=C/CC/C(=C/COC1=CC2=C(C=C1)C=CC(=O)O2)/C)/C)C",
    "umbelliferone": r"C1=CC(=CC2=C1C=CC(=O)O2)O",
    "scopoletin": r"COC1=C(C=C2C(=C1)C=CC(=O)O2)O",
    "imperatorin": r"CC(=CCOC1=C2C(=CC3=C1OC=C3)C=CC(=O)O2)C",
    "herniarin": r"COC1=CC2=C(C=C1)C=CC(=O)O2",
    "farnesiferola": r"C[C@@]12CC[C@H](C([C@H]1CCC(=C)[C@@H]2COC3=CC4=C(C=C3)C=CC(=O)O4)(C)C)O",
    "farnesiferolb": r"C/C(=C\COC1=CC2=C(C=C1)C=CC(=O)O2)/CC[C@@H]3C(=C)CC[C@@H](C3(C)C)O",
    "farnesiferolc": r"C/C(=C\COC1=CC2=C(C=C1)C=CC(=O)O2)/CC[C@H]3[C@@]4(CC[C@H](C3(C)C)O4)C",
}

# The first obvious feature to add will be the Molecular Weights (From PubChem)
MWs = {
    "auraptene": 298.4,
    "osthole": 244.28,
    "esculetin": 178.14,
    "galbanicacid": 398.5,
    "umbelliprenin": 366.5,
    "umbelliferone": 162.14,
    "scopoletin": 192.17,
    "imperatorin": 270.28,
    "herniarin": 176.17,
    "farnesiferola": 382.5,
    "farnesiferolb": 382.5,
    "farnesiferolc": 382.5,
}

# The second Important Feature will be LogP (XLogP3 From PubChem)
LogPs = {
    "auraptene": 5.3,
    "osthole": 3.8,
    "esculetin": 1.2,
    "galbanicacid": 5.1,
    "umbelliprenin": 7.1,
    "umbelliferone": 1.6,
    "scopoletin": 1.5,
    "imperatorin": 3.4,
    "herniarin": 1.9,
    "farnesiferola": 4.9,
    "farnesiferolb": 5.3,
    "farnesiferolc": 5.5,
}

# Number of the Hydrogen bond donors
numHDonors = {
    "auraptene": 0,
    "osthole": 0,
    "esculetin": 2,
    "galbanicacid": 1,
    "umbelliprenin": 1,
    "umbelliferone": 0,
    "scopoletin": 1,
    "imperatorin": 0,
    "herniarin": 0,
    "farnesiferola": 1,
    "farnesiferolb": 1,
    "farnesiferolc": 0,
}

# Number of the Hydrogen bond acceptors
numHAcceptors = {
    "auraptene": 3,
    "osthole": 3,
    "esculetin": 4,
    "galbanicacid": 5,
    "umbelliprenin": 3,
    "umbelliferone": 3,
    "scopoletin": 4,
    "imperatorin": 4,
    "herniarin": 3,
    "farnesiferola": 4,
    "farnesiferolb": 4,
    "farnesiferolc": 4,
}

# Number of rotatable bonds
rotatableBonds = {
    "auraptene": 6,
    "osthole": 3,
    "esculetin": 0,
    "galbanicacid": 6,
    "umbelliprenin": 9,
    "umbelliferone": 0,
    "scopoletin": 1,
    "imperatorin": 3,
    "herniarin": 1,
    "farnesiferola": 3,
    "farnesiferolb": 6,
    "farnesiferolc": 6,
}

# Topological Polar Surface Area (From PubChem)
TPSAs = {
    "auraptene": 35.5,
    "osthole": 35.5,
    "esculetin": 66.8,
    "galbanicacid": 72.8,
    "umbelliprenin": 35.5,
    "umbelliferone": 46.5,
    "scopoletin": 55.8,
    "imperatorin": 48.7,
    "herniarin": 35.5,
    "farnesiferola": 55.8,
    "farnesiferolb": 55.8,
    "farnesiferolc": 44.8,
}

# Ring Counts
ringCounts = {
    "auraptene": 2,
    "osthole": 2,
    "esculetin": 2,
    "galbanicacid": 3,
    "umbelliprenin": 2,
    "umbelliferone": 2,
    "scopoletin": 2,
    "imperatorin": 2,
    "herniarin": 2,
    "farnesiferola": 4,
    "farnesiferolb": 3,
    "farnesiferolc": 2,
}

# FractionCsp3: The fraction of carbon atoms that are sp3 hybridized. (From SwissAdme.ch)
fCsp3s = {
    "auraptene": 0.32,
    "osthole": 0.27,
    "esculetin": 0,
    "galbanicacid": 0.5,
    "umbelliprenin": 0.38,
    "umbelliferone": 0,
    "scopoletin": 0.10,
    "imperatorin": 0.19,
    "herniarin": 0.10,
    "farnesiferola": 0.54,
    "farnesiferolb": 0.46,
    "farnesiferolc": 0.54,
}

# Morgan fingerprint settings
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)

# Read the data
data1 = pd.read_csv("Data/Data.csv")
data2 = pd.read_csv("Data/new_data.csv")
data3 = pd.read_csv("Data/FarnesiferolData.csv")

# Fix time in data2 and data3
data2["Time"] = data2["Time"].str.replace("h", "", regex=False).astype(float)
data3["Time"] = data3["Time"].str.replace("*", "", regex=False).astype(float)

# Add the year feature to the author feature in data3
data3["Author"] = data3["Author"] + " " + data3["Year"].astype('Int64').astype(str)

# Pick the required features
data2 = data2[["Author", "Cancer", "CellLine", "Coumarin", "Dose", "Time", "Viability"]]
data3 = data3[["Author", "Cancer", "CellLine", "Coumarin", "Dose", "Time", "Viability"]]

# Drop null columns
data1 = data1.dropna(axis=1)

# Drop null rows:
data1 = data1.dropna(axis=0)
data2 = data2.dropna(axis=0)
data3 = data3.dropna(axis=0)

# Concatenate the datasets
data = pd.concat(objs=[data1, data2, data3], ignore_index=True)

# Print the sizes of the datasets
print()
print("Size of the first dataset:", data1.__len__())
print("Size of the second dataset:", data2.__len__())
print("Size of the third dataset:", data3.__len__())
print("Size of the combined dataset:", data.__len__())

line = "=" * 100 + "\n"
print(line)

# Coumarins in each dataset
print("Data1 coumarins:")
print(data1["Coumarin"].value_counts())
print()

print("Data2 coumarins:")
print(data2["Coumarin"].value_counts())
print(line)

print("Data3 coumarins:")
print(data3["Coumarin"].value_counts())
print(line)

# Remove whitespace from the feature names
data.columns = data.columns.str.replace(r"\s+", "", regex=True)

# Remove whitespace from categorical values and make them lowercase
data["Coumarin"] = data["Coumarin"].str.replace(r"\s+", "", regex=True)
data["Coumarin"] = data["Coumarin"].str.lower()

data["Cancer"] = data["Cancer"].str.replace(r"\s+", "", regex=True)
data["Cancer"] = data["Cancer"].str.lower()

data["CellLine"] = data["CellLine"].str.replace(r"\s+", "", regex=True)
data["CellLine"] = data["CellLine"].str.lower()

# Start cleaning of the categorical values

# Normalize string values
def normalize_string(value):
    if pd.isna(value):
        return value

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Normalize all common Unicode dash/hyphen characters
    value = (
        value.replace("‐", "-")
        .replace("‒", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("―", "-")
    )

    # Normalize whitespace
    value = " ".join(value.split())

    # Lowercase
    value = value.lower().strip()

    return value

# Normalize the categorical (string) features
for column in ["Cancer", "CellLine", "Coumarin"]:
    data[column] = data[column].map(normalize_string)

# Fix cases where the same cancer has different names
cancer_aliases = {
    "bbladder": "bladder",
    "cervical*": "cervical",
    "cervicalcarcinoma": "cervical",
    "melanoma*": "melanoma",
    "ovariancancer": "ovarian",
    "larynx": "laryngeal",
    "renal": "kidney",
    "cholangio": "cholangiocarcinoma",
}

# Log cancer aliases before applying them
print("\n" + "=" * 100)
print("CANCER ALIAS CHANGES")
print("=" * 100)

for old_value, new_value in cancer_aliases.items():
    count = (data["Cancer"] == old_value).sum()

    if count > 0:
        print(f"{old_value:<25} -> {new_value:<25} : {count} rows")

data["Cancer"] = data["Cancer"].replace(cancer_aliases)

# Same fix for the cell lines
cell_line_aliases = {
    "pc3": "pc-3",
    "ht29": "ht-29",
    "ht-29": "ht-29",
    "hct116": "hct-116",
    "hepg-2": "hepg2",
    "hcclm3": "hcc-lm3",
    "ovcar3": "ovcar-3",
    "h1299": "nci-h1299",
    "jurkata": "jurkat",
    "jurkatt": "jurkat",
}

# Log cell-line aliases before applying them
print("\n" + "=" * 100)
print("CELL-LINE ALIAS CHANGES")
print("=" * 100)

for old_value, new_value in cell_line_aliases.items():
    count = (data["CellLine"] == old_value).sum()

    if count > 0:
        print(f"{old_value:<25} -> {new_value:<25} : {count} rows")

data["CellLine"] = data["CellLine"].replace(cell_line_aliases)

# Remove exact duplicate rows
duplicate_count = data.duplicated().sum()

print()
print(f"Exact duplicate rows found: {duplicate_count}")

data = data.drop_duplicates().reset_index(drop=True)

# Check for missing values after categorical cleaning
print("\n" + "=" * 100)
print("MISSING VALUES AFTER CLEANING")
print("=" * 100)
print(data.isna().sum())

# Check for non-finite numerical values
numeric_columns = [
    "Dose",
    "Time",
    "Viability",
]

print("\n" + "=" * 100)
print("NON-FINITE NUMERICAL VALUES")
print("=" * 100)

print("NaN values:")
print(data[numeric_columns].isna().sum())
print()

print("Infinite values:")
print(
    data[numeric_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
)

# Check for impossible numerical values
print("\n" + "=" * 100)
print("NUMERICAL VALUE VALIDATION")
print("=" * 100)

print("Negative Dose values:", (data["Dose"] < 0).sum())
print("Negative Time values:", (data["Time"] < 0).sum())
print("Viability below 0:", (data["Viability"] < 0).sum())
print("Viability above 100:", (data["Viability"] > 100).sum())

print()
print("Rows with invalid Dose values:")
print(data[data["Dose"] < 0])

print()
print("Rows with invalid Time values:")
print(data[data["Time"] < 0])

print()
print("Rows with Viability outside [0, 100]:")
print(data[(data["Viability"] < 0) | (data["Viability"] > 100)])

# Check that every coumarin has all required molecular descriptors
print("\n" + "=" * 100)
print("COUMARIN MAPPING CHECK")
print("=" * 100)

print("Unknown coumarins:")

unknown_coumarins = data.loc[
    ~data["Coumarin"].isin(MWs.keys()),
    "Coumarin",
].unique()

print(unknown_coumarins)

if len(unknown_coumarins) > 0:
    raise ValueError(
        "Unknown coumarin names were found. "
        "Check the Coumarin values and descriptor dictionaries."
    )

# Check that every coumarin has a valid SMILES
unknown_smiles = data.loc[
    ~data["Coumarin"].isin(SMILES.keys()),
    "Coumarin",
].unique()

if len(unknown_smiles) > 0:
    raise ValueError(
        "One or more coumarins do not have a SMILES entry. "
        "Check the SMILES dictionary."
    )

# Check for contradictory experimental conditions
duplicate_key_columns = [
    "Author",
    "Cancer",
    "CellLine",
    "Coumarin",
    "Dose",
    "Time",
]

condition_groups = (
    data.groupby(duplicate_key_columns)["Viability"]
    .agg(
        ViabilityValues=list,
        NumberOfDifferentViabilityValues="nunique",
        NumberOfRows="size",
    )
    .reset_index()
)

conflicting_rows = condition_groups[
    condition_groups["NumberOfDifferentViabilityValues"] > 1
]

print("\n" + "=" * 100)
print("CONFLICTING EXPERIMENTAL CONDITIONS")
print("=" * 100)

print(
    "Number of condition groups with multiple Viability values:",
    len(conflicting_rows),
)

print()

if len(conflicting_rows) > 0:
    print(
        conflicting_rows[
            duplicate_key_columns
            + [
                "ViabilityValues",
                "NumberOfDifferentViabilityValues",
                "NumberOfRows",
            ]
        ].to_string(index=False)
    )
else:
    print("No conflicting experimental conditions were found.")

# Check for duplicate rows while ignoring Author
duplicate_columns_without_author = [
    "Cancer",
    "CellLine",
    "Coumarin",
    "Dose",
    "Time",
    "Viability",
]

duplicate_rows_without_author = (
    data.groupby(duplicate_columns_without_author)
    .agg(
        Authors=("Author", list),
        NumberOfAuthors=("Author", "nunique"),
        NumberOfRows=("Author", "size"),
    )
    .reset_index()
)

duplicate_rows_without_author = duplicate_rows_without_author[
    duplicate_rows_without_author["NumberOfAuthors"] > 1
]

print("\n" + "=" * 100)
print("DUPLICATE ROWS IGNORING AUTHOR")
print("=" * 100)

print(
    "Number of duplicate groups ignoring Author:",
    len(duplicate_rows_without_author),
)

print()

if len(duplicate_rows_without_author) > 0:
    print(
        duplicate_rows_without_author[
            duplicate_columns_without_author
            + [
                "Authors",
                "NumberOfAuthors",
                "NumberOfRows",
            ]
        ].to_string(index=False)
    )
else:
    print("No duplicate rows were found when ignoring Author.")


# Check cell lines associated with multiple cancer categories
cellline_cancer_counts = (
    data.groupby("CellLine")["Cancer"]
    .nunique()
    .reset_index(name="NumberOfCancerCategories")
)

multi_cancer_celllines = cellline_cancer_counts[
    cellline_cancer_counts["NumberOfCancerCategories"] > 1
]

print("\n" + "=" * 100)
print("CELL-LINES ASSOCIATED WITH MULTIPLE CANCER CATEGORIES")
print("=" * 100)

print(
    "Number of cell lines associated with multiple cancer categories:",
    multi_cancer_celllines.__len__(),
)

print()

if multi_cancer_celllines.__len__() > 0:
    for cell_line in multi_cancer_celllines["CellLine"]:
        cancers = sorted(
            data.loc[data["CellLine"] == cell_line, "Cancer"].unique()
        )
        print(f"{cell_line:<20} -> {', '.join(cancers)}")
else:
    print("No cell lines are associated with multiple cancer categories.")

# Log the cleaned features and data shape
print("\n" + "=" * 100)
print("CLEANED CANCER VALUES")
print("=" * 100)
print(data["Cancer"].value_counts().sort_index())

print("\n" + "=" * 100)
print("CLEANED CELL-LINE VALUES")
print("=" * 100)
print(data["CellLine"].value_counts().sort_index())

print("\n" + "=" * 100)
print("DATASET SHAPE")
print("=" * 100)
print(data.shape)

# Add the molecular features
data["MW"] = data["Coumarin"].map(MWs)
data["LogP"] = data["Coumarin"].map(LogPs)
data["NumHDonors"] = data["Coumarin"].map(numHDonors)
data["NumHAcceptors"] = data["Coumarin"].map(numHAcceptors)
data["RotatableBonds"] = data["Coumarin"].map(rotatableBonds)
data["TPSA"] = data["Coumarin"].map(TPSAs)
data["RingCounts"] = data["Coumarin"].map(ringCounts)
data["fCsp3"] = data["Coumarin"].map(fCsp3s)

# Add LogDose to the dataset
data["LogDose"] = np.log1p(data["Dose"])
print(data["LogDose"].head())

# Validate the engineered molecular features
molecular_features = [
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "RotatableBonds",
    "TPSA",
    "RingCounts",
    "fCsp3",
]

print("\n" + "=" * 100)
print("STRUCTURAL FEATURES VALIDATION")
print("=" * 100)
print(data[molecular_features].isna().sum())

if data[molecular_features].isna().any().any():
    raise ValueError(
        "One or more molecular descriptors could not be mapped. "
        "Check the Coumarin values and descriptor dictionaries."
    )
else:
    print("No problems with the structural features.")

# Check descriptor consistency within each Coumarin
descriptor_check = (
    data.groupby("Coumarin")[molecular_features]
    .nunique()
)

print("\n" + "=" * 100)
print("STRUCTURAL FEATURE CONSISTENCY")
print("=" * 100)
print(descriptor_check)

if (descriptor_check > 1).any().any():
    raise ValueError(
        "A Coumarin has inconsistent molecular descriptor values."
    )
else:
    print("All molecular descriptors are consistent within each Coumarin.")

# Generate Morgan fingerprints
def generate_fingerprint(coumarin):
    smiles = SMILES[coumarin]
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"RDKit could not parse the SMILES for coumarin: {coumarin}"
        )

    fingerprint = fingerprint_generator.GetFingerprint(molecule)

    return [int(bit) for bit in fingerprint.ToBitString()]

# Validate all SMILES before generating fingerprints
print("\n" + "=" * 100)
print("SMILES VALIDATION")
print("=" * 100)

for coumarin, smiles in SMILES.items():
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"Invalid SMILES detected for coumarin: {coumarin}"
        )

    print(f"{coumarin:<20} -> valid")

# Generate fingerprints
fingerprint_columns = [
    f"Morgan_{i}"
    for i in range(MORGAN_BITS)
]

fingerprints = data["Coumarin"].map(generate_fingerprint)

fingerprint_data = pd.DataFrame(
    fingerprints.tolist(),
    columns=fingerprint_columns,
    index=data.index,
)

# Add fingerprints to the dataset
data = pd.concat([data, fingerprint_data], axis=1)

# Validate fingerprints
print("\n" + "=" * 100)
print("MORGAN FINGERPRINT VALIDATION")
print("=" * 100)

print("Fingerprint radius:", MORGAN_RADIUS)
print("Fingerprint size:", MORGAN_BITS)
print("Number of fingerprint columns:", len(fingerprint_columns))

if len(fingerprint_columns) != MORGAN_BITS:
    raise ValueError("Incorrect number of fingerprint columns.")

if data[fingerprint_columns].isna().any().any():
    raise ValueError("NaN values were found in the Morgan fingerprints.")

if not data[fingerprint_columns].isin([0, 1]).all().all():
    raise ValueError("Morgan fingerprints contain values other than 0 and 1.")

print("All Morgan fingerprints are valid.")

# Check fingerprint consistency within each Coumarin
fingerprint_check = (
    data.groupby("Coumarin")[fingerprint_columns]
    .nunique()
    .max(axis=1)
)

print("\nFingerprint consistency within each Coumarin:")

if (fingerprint_check > 1).any():
    raise ValueError(
        "A Coumarin has inconsistent Morgan fingerprints."
    )
else:
    print("All Morgan fingerprints are consistent within each Coumarin.")

# Take a look at the new data structure and columns
print()
print("Head of the data after new features:")
print(data.head())
print(line)

# Remove Morgan FingerPrints that are all zero across every coumarin
morgan_cols = data.columns[data.columns.str.startswith("Morgan_")]
all_zero_morgan = morgan_cols[(data[morgan_cols] == 0).all(axis=0)]

print(f"All-zero Morgan features ({len(all_zero_morgan)}):")
data = data.drop(columns=all_zero_morgan)
print("The all zero Morgan have been dropped!")

# Check the correlation matrix of the molecular descriptors
cols = [
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "RotatableBonds",
    "TPSA",
    "RingCounts",
    "fCsp3",
]


corr_matrix = data[cols].corr()

print()
print("\nCorrelation Matrix of the newly engineered features:")
print(tabulate(corr_matrix, headers="keys"))

# Create heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="Spectral",
    center=0,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)

plt.title("Structural Features Correlation Matrix", fontsize=18)
plt.tight_layout()
plt.show()
plt.savefig("Figures/Fig1_Correlation_Matrix.png")
plt.close()

# Save the processed data
data.to_csv("Data/Processed_Data.csv", index=False)

print()
print("Processed data saved to: Data/Processed_Data.csv")

# Make classification for 2 classes
binary_classification_data = data.copy()

binary_classification_data["Viability"] = (
(binary_classification_data["Viability"] - 50).abs() <= 10
).astype(int)

print()
print(line)

print("Binary Classification - Target Distribution:")
print(
binary_classification_data["Viability"]
.value_counts()
.sort_index()
)

print()
print("Binary Classification - Target Proportions:")
print(
binary_classification_data["Viability"]
.value_counts(normalize=True)
.sort_index()
)

print()
print("Head of binary classification data:")
print(tabulate(
    pd.DataFrame(binary_classification_data["Viability"]).head(),
    headers='keys',
    tablefmt="rounded_grid"
    ))

binary_classification_data.to_csv(
"Data/Binary_Classification_Data.csv",
index=False,
)

print()
print("Binary classification data saved to: Data/Binary_Classification_Data.csv")

# Make 5 class classification data:
five_class_classification_data = data.copy()

five_class_classification_data["Viability"] = pd.cut(
five_class_classification_data["Viability"],
bins=[-np.inf, 20, 40, 60, 80, np.inf],
labels=[1, 2, 3, 4, 5],
right=True,
).astype("Int64")

print()
print(line)

print("Five-Class Classification - Target Distribution:")
print(
five_class_classification_data["Viability"]
.value_counts()
.sort_index()
)

print()
print("Five-Class Classification - Target Proportions:")
print(
five_class_classification_data["Viability"]
.value_counts(normalize=True)
.sort_index()
)

print()
print("Head of 5 class classification data:")
print(tabulate(
    pd.DataFrame(five_class_classification_data["Viability"]).head(),
    headers='keys',
    tablefmt="rounded_grid"
    ))

five_class_classification_data.to_csv(
"Data/Five_Class_Classification_Data.csv",
index=False,
)

print()
print(
"Five-class classification data saved to: "
"Data/Five_Class_Classification_Data.csv"
)

# ------------------------------------------------------------
# PCA EXPLAINED VARIANCE ANALYSIS
# ------------------------------------------------------------

PCA_OUTPUT_DIRECTORY = Path("Figures/PCA_ExplainedVariance")
PCA_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

morgan_features = list(
    data.columns[data.columns.str.startswith("Morgan_")]
)
experimental_features = ["LogDose", "Time"]

pca_feature_configurations = {
    "PCA_All_Numerical": (
        molecular_features + experimental_features + morgan_features
    ),
    "PCA_Descriptors_Fingerprints": (
        molecular_features + morgan_features
    ),
    "PCA_Fingerprints": morgan_features,
}

pca_results = {}

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(20, 11),
)

print("\n" + "=" * 100)
print("PCA EXPLAINED VARIANCE ANALYSIS")
print("=" * 100)

for column, (configuration, features) in enumerate(
    pca_feature_configurations.items()
):

    print("\n" + "-" * 100)
    print(configuration)
    print("-" * 100)

    X = data[features].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(svd_solver="full")
    pca.fit(X_scaled)

    individual_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(individual_variance)

    eigenvalues = pca.explained_variance_
    variance_tolerance = eigenvalues.max() * 1e-12

    effective_components = np.count_nonzero(
        eigenvalues > variance_tolerance
    )

    pca_results[configuration] = {
        "NumberOfFeatures": len(features),
        "NumberOfSamples": len(X),
        "EffectiveComponents": effective_components,
        "IndividualVariance": individual_variance,
        "CumulativeVariance": cumulative_variance,
    }

    print("Number of original features:", len(features))
    print("Number of samples:", len(X))
    print("Maximum meaningful components:", effective_components)

    print("\nCumulative explained variance:")

    for threshold in [0.80, 0.90, 0.95, 0.99, 1.00]:

        components_required = np.searchsorted(
            cumulative_variance,
            threshold,
        ) + 1

        if components_required <= effective_components:
            print(
                f"{threshold:.0%} variance: "
                f"{components_required} components"
            )
        else:
            print(f"{threshold:.0%} variance: Not reached")

    component_numbers = np.arange(1, len(individual_variance) + 1)

    # --------------------------------------------------------
    # FIRST ROW: CUMULATIVE EXPLAINED VARIANCE
    # --------------------------------------------------------

    ax = axes[0, column]

    ax.plot(
        component_numbers,
        cumulative_variance,
        marker="o",
        markersize=3,
    )

    ax.axvline(
        effective_components,
        linestyle="--",
        label=f"Effective components: {effective_components}",
    )

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        ax.axhline(
            threshold,
            linestyle=":",
            alpha=0.7,
            label=f"{threshold:.0%} variance",
        )

    ax.set_title(configuration, fontsize=12)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --------------------------------------------------------
    # SECOND ROW: INDIVIDUAL EXPLAINED VARIANCE
    # --------------------------------------------------------

    ax = axes[1, column]

    ax.plot(
        component_numbers,
        individual_variance,
        color="darkorange",
        marker="o",
        markersize=3,
    )

    ax.axvline(
        effective_components,
        color="black",
        linestyle="--",
        label=f"Effective components: {effective_components}",
    )

    ax.set_title(f"{configuration} - Individual", fontsize=12)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Individual Explained Variance")
    ax.set_xlim(0, 32)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# ------------------------------------------------------------
# FINALIZE AND SAVE FIGURE
# ------------------------------------------------------------

fig.suptitle(
    "PCA Explained Variance Analysis",
    fontsize=17,
    fontweight="bold",
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    PCA_OUTPUT_DIRECTORY / "PCA_Explained_Variance_Comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

pca_summary = pd.DataFrame.from_dict(
    {
        configuration: {
            "Original Features": result["NumberOfFeatures"],
            "Samples": result["NumberOfSamples"],
            "Effective Components": result["EffectiveComponents"],
        }
        for configuration, result in pca_results.items()
    },
    orient="index",
)

print("\n" + "=" * 100)
print("PCA SUMMARY")
print("=" * 100)

print(tabulate(pca_summary, headers="keys", tablefmt="rounded_grid"))

print("\nPCA plot saved to:", PCA_OUTPUT_DIRECTORY)

# ------------------------------------------------------------
# PCA COMPONENT AVAILABILITY: LOCO AND LOGO
# ------------------------------------------------------------

from sklearn.model_selection import LeaveOneGroupOut

PCA_GROUPS = {
    "Group_1": [
        "auraptene",
        "umbelliprenin",
        "imperatorin",
    ],
    "Group_2": [
        "umbelliferone",
        "herniarin",
        "scopoletin",
        "esculetin",
        "osthole",
    ],
    "Group_3": [
        "galbanicacid",
        "farnesiferola",
        "farnesiferolb",
        "farnesiferolc",
    ],
}

PCA_COMPONENT_LIMITS = {
    "PCA_All_Numerical": 13,
    "PCA_Descriptors_Fingerprints": 11,
    "PCA_Fingerprints": 11,
}


def get_effective_pca_components(X):
    """
    Calculate the numerical rank of standardized training features.

    The tolerance is consistent with the previous exploratory
    PCA analysis.
    """

    X = X.astype(float)

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(svd_solver="full")
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_

    if len(eigenvalues) == 0 or eigenvalues.max() == 0:
        return 0

    tolerance = eigenvalues.max() * 1e-12

    return int(np.count_nonzero(eigenvalues > tolerance))


def analyze_pca_fold_components(
    experiment_name,
    experiment_data,
    groups,
):
    """
    Calculate the effective PCA component count for every
    training fold of an experiment.
    """

    print("\n" + "=" * 110)
    print(f"{experiment_name} - PCA COMPONENT AVAILABILITY")
    print("=" * 110)

    fold_results = []

    logo = LeaveOneGroupOut()

    for fold, (train_indices, test_indices) in enumerate(
        logo.split(experiment_data, groups=groups),
        start=1,
    ):

        train_data = experiment_data.iloc[train_indices]

        left_out_groups = sorted(
            experiment_data.iloc[test_indices]["Coumarin"].unique()
        )

        left_out_label = ", ".join(left_out_groups)

        print("\n" + "-" * 100)
        print(f"Fold {fold}: Left out -> {left_out_label}")
        print("Training samples:", len(train_data))
        print("-" * 100)

        fold_result = {
            "Fold": fold,
            "LeftOut": left_out_label,
            "TrainingSamples": len(train_data),
        }

        for configuration, features in pca_feature_configurations.items():

            X_train = train_data[features]

            effective_components = get_effective_pca_components(
                X_train
            )

            maximum_allowed = PCA_COMPONENT_LIMITS[configuration]

            valid_components = min(
                effective_components,
                maximum_allowed,
            )

            fold_result[configuration] = effective_components

            print(
                f"{configuration:<35}"
                f"Effective: {effective_components:<5}"
                f"Maximum allowed: {valid_components}"
            )

        fold_results.append(fold_result)

    results = pd.DataFrame(fold_results)

    print("\n" + "-" * 110)
    print(f"{experiment_name} SUMMARY")
    print("-" * 110)

    print(
        tabulate(
            results,
            headers="keys",
            tablefmt="rounded_grid",
            showindex=False,
        )
    )

    print("\nMinimum available components across folds:")

    for configuration in pca_feature_configurations:

        minimum_components = results[configuration].min()

        maximum_allowed = PCA_COMPONENT_LIMITS[configuration]

        print(
            f"{configuration:<35}"
            f"Minimum effective rank: {minimum_components:<5}"
            f"Candidate range: 1-{min(minimum_components, maximum_allowed)}"
        )

    return results


# ------------------------------------------------------------
# LOCO
# ------------------------------------------------------------

loco_pca_results = analyze_pca_fold_components(
    experiment_name="LOCO",
    experiment_data=data.reset_index(drop=True),
    groups=data["Coumarin"].reset_index(drop=True),
)


# ------------------------------------------------------------
# LOGO
# ------------------------------------------------------------

logo_data = data[
    data["Coumarin"].isin(
        PCA_GROUPS["Group_1"]
        + PCA_GROUPS["Group_2"]
        + PCA_GROUPS["Group_3"]
    )
].copy().reset_index(drop=True)

logo_data["CoumarinGroup"] = logo_data["Coumarin"].map(
    {
        coumarin: group_name
        for group_name, coumarins in PCA_GROUPS.items()
        for coumarin in coumarins
    }
)

logo_pca_results = analyze_pca_fold_components(
    experiment_name="LOGO",
    experiment_data=logo_data,
    groups=logo_data["CoumarinGroup"],
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

PCA_COMPONENT_OUTPUT_DIRECTORY = Path(
    "Figures/PCA_Component_Availability"
)

PCA_COMPONENT_OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

loco_pca_results.to_csv(
    PCA_COMPONENT_OUTPUT_DIRECTORY / "LOCO_PCA_Components.csv",
    index=False,
)

logo_pca_results.to_csv(
    PCA_COMPONENT_OUTPUT_DIRECTORY / "LOGO_PCA_Components.csv",
    index=False,
)

print("\nLOCO and LOGO PCA component results saved to:")
print(PCA_COMPONENT_OUTPUT_DIRECTORY)