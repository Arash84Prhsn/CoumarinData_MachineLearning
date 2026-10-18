import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from tabulate import tabulate


DATA_PATH = "Data/Processed_Data.csv"

TRAIN_COUMARINS = [
    "auraptene",
    "esculetin",
    "galbanicacid",
    "herniarin",
    "osthole",
    "umbelliferone",
    "umbelliprenin",
]

EXTERNAL_COUMARINS = [
    "scopoletin",
    "imperatorin",
]

MOLECULAR_FEATURES = [
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "RingCounts",
    "fCsp3",
]


data = pd.read_csv(DATA_PATH)

compound_data = (
    data[
        data["Coumarin"].isin(
            TRAIN_COUMARINS + EXTERNAL_COUMARINS
        )
    ]
    .groupby("Coumarin")[MOLECULAR_FEATURES]
    .first()
    .loc[TRAIN_COUMARINS + EXTERNAL_COUMARINS]
)


print("=" * 100)
print("MOLECULAR DESCRIPTOR SPACE ANALYSIS")
print("=" * 100)

print("\nCompound descriptor table:")
print(
    tabulate(
        compound_data.reset_index(),
        headers="keys",
        floatfmt=".4f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)


scaler = StandardScaler()

scaled_descriptors = scaler.fit_transform(
    compound_data[MOLECULAR_FEATURES]
)

scaled_descriptors = pd.DataFrame(
    scaled_descriptors,
    index=compound_data.index,
    columns=MOLECULAR_FEATURES,
)


training_descriptors = scaled_descriptors.loc[
    TRAIN_COUMARINS
]

external_descriptors = scaled_descriptors.loc[
    EXTERNAL_COUMARINS
]


distance_matrix = pairwise_distances(
    external_descriptors,
    training_descriptors,
    metric="euclidean",
)


distance_table = pd.DataFrame(
    distance_matrix,
    index=EXTERNAL_COUMARINS,
    columns=TRAIN_COUMARINS,
)


print("\n")
print("=" * 100)
print("STANDARDIZED EUCLIDEAN DISTANCES")
print("=" * 100)

print(
    tabulate(
        distance_table.reset_index(),
        headers="keys",
        floatfmt=".4f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)


print("\n")
print("=" * 100)
print("NEAREST TRAINING COUMARINS")
print("=" * 100)


for external in EXTERNAL_COUMARINS:
    nearest = (
        distance_table.loc[external]
        .sort_values()
    )

    print(f"\n{external}:")
    
    for compound, distance in nearest.items():
        print(
            f"  {compound:<18} "
            f"distance = {distance:.4f}"
        )


print("\n")
print("=" * 100)
print("NEAREST-COMPOUND SUMMARY")
print("=" * 100)


summary = []

for external in EXTERNAL_COUMARINS:
    distances = (
        distance_table.loc[external]
        .sort_values()
    )

    nearest_compound = distances.index[0]
    nearest_distance = distances.iloc[0]

    second_compound = distances.index[1]
    second_distance = distances.iloc[1]

    summary.append({
        "External Coumarin": external,
        "Nearest": nearest_compound,
        "Distance": nearest_distance,
        "2nd Nearest": second_compound,
        "2nd Distance": second_distance,
    })


summary = pd.DataFrame(summary)

print(
    tabulate(
        summary,
        headers="keys",
        floatfmt=".4f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity


MORGAN_RADIUS = 2
MORGAN_BITS = 2048


SMILES = {
    "auraptene": "CC(=CCC/C(=C/COC1=CC2=C(C=C1)C=CC(=O)O2)/C)C",
    "osthole": "CC(=CCC1=C(C=CC2=C1OC(=O)C=C2)OC)C",
    "esculetin": "C1=CC(=O)OC2=CC(=C(C=C21)O)O",
    "galbanicacid": "C[C@H]1CCC(=C(C)C)[C@@H]([C@@]1(C)COC2=CC3=C(C=C2)C=CC(=O)O3)CCC(=O)O",
    "umbelliprenin": "CC(=CCC/C(=C/CC/C(=C/COC1=CC2=C(C=C1)C=CC(=O)O2)/C)/C)C",
    "umbelliferone": "C1=CC(=CC2=C1C=CC(=O)O2)O",
    "scopoletin": "COC1=C(C=C2C(=C1)C=CC(=O)O2)O",
    "imperatorin": "CC(=CCOC1=C2C(=CC3=C1OC=C3)C=CC(=O)O2)C",
    "herniarin": "COC1=CC2=C(C=C1)C=CC(=O)O2",
}


generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)


fingerprints = {}

for compound, smiles in SMILES.items():
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"Could not parse SMILES for {compound}"
        )

    fingerprints[compound] = generator.GetFingerprint(
        molecule
    )


similarity_matrix = pd.DataFrame(
    index=EXTERNAL_COUMARINS,
    columns=TRAIN_COUMARINS,
    dtype=float,
)


for external in EXTERNAL_COUMARINS:
    for training in TRAIN_COUMARINS:
        similarity_matrix.loc[
            external,
            training
        ] = TanimotoSimilarity(
            fingerprints[external],
            fingerprints[training],
        )


print("\n")
print("=" * 100)
print("MORGAN FINGERPRINT TANIMOTO SIMILARITY")
print("=" * 100)

print(
    tabulate(
        similarity_matrix.reset_index(),
        headers="keys",
        floatfmt=".4f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)


print("\n")
print("=" * 100)
print("NEAREST TRAINING COUMARINS BY MORGAN SIMILARITY")
print("=" * 100)


for external in EXTERNAL_COUMARINS:
    similarities = (
        similarity_matrix.loc[external]
        .sort_values(
            ascending=False
        )
    )

    print(f"\n{external}:")

    for compound, similarity in similarities.items():
        print(
            f"  {compound:<18} "
            f"Tanimoto = {similarity:.4f}"
        )


morgan_summary = []

for external in EXTERNAL_COUMARINS:
    similarities = (
        similarity_matrix.loc[external]
        .sort_values(
            ascending=False
        )
    )

    nearest_compound = similarities.index[0]
    nearest_similarity = similarities.iloc[0]

    second_compound = similarities.index[1]
    second_similarity = similarities.iloc[1]

    morgan_summary.append({
        "External Coumarin": external,
        "Nearest": nearest_compound,
        "Similarity": nearest_similarity,
        "2nd Nearest": second_compound,
        "2nd Similarity": second_similarity,
    })


morgan_summary = pd.DataFrame(
    morgan_summary
)


print("\n")
print("=" * 100)
print("MORGAN NEAREST-COMPOUND SUMMARY")
print("=" * 100)

print(
    tabulate(
        morgan_summary,
        headers="keys",
        floatfmt=".4f",
        tablefmt="rounded_outline",
        showindex=False,
    )
)