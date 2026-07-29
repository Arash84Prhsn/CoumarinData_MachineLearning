import pandas as pd;
import numpy as np;

data = pd.read_excel("Data/meta Niloofar 8.10.404.final.xlsx");
data = data.drop(columns=['Column1', 'N0.']);

# Forward fill these categorical features
data["Author"] = data["Author"].ffill();
data["Cancer"] = data["Cancer"].ffill();
data["Cell Line"] = data["Cell Line"].ffill();

# 1. Remove all whitespace
data["Unit"] = data["Unit"].str.replace(r"\s+", "", regex=True);
data["Time"] = data["Time"].str.replace(r"\s+", "", regex=True);

# 2. Normalize unicode µ signs to standard 'µ'
data["Unit"] = data["Unit"].str.replace('μ', 'µ')

# 3. Lowercase for consistency
data["Unit"] = data["Unit"].str.lower();
data["Time"] = data["Time"].str.lower();


print(data["Unit"].unique())
print(data["Unit"].value_counts())

# Molecular weight of Osthole.
MW = 244.28;

# The new data (Osthole data) needs unit transformation.
def to_uM(row):
    if row["Unit"].strip() == "mg/ml":
        return row["Dose"] * 1e6 / MW;

    elif row["Unit"].strip() == "nm":
        return row["Dose"] * 1e-3;

    elif row["Unit"].strip() == "µg/ml":
        return row['Dose'] * 1e3 / MW;

    else:
        return row["Dose"];

data["Dose"] = data.apply(to_uM, axis=1)
data["Dose"] = data["Dose"].round(2);

data["Unit"] = "µM"
data["Coumarin"] = "Osthole"
col = data.pop("Coumarin");
data.insert(3, 'Coumarin', col);
col = data.pop("Time");
data.insert(5, 'Time', col);

data.to_csv("Data/cleanedData.csv", index=False);

# Now we shall concatenate this data to the old one.

# Load the old dataset

oldData = pd.read_csv("Data/oldData.csv");

# print(oldData.columns);

# Just some re ordering and renaming of the feaatures to concat to the new osthole data easily
oldData = oldData[['Seo', 'Cancer Type', 'Cell Line', 'Coumarin Type', 'Coumarin Dose', 'Time', 'Viability']];
oldData.columns = ['Author', 'Cancer', 'Cell Line', 'Coumarin', 'Dose', 'Time', "Viability"];
newData = data[['Author', 'Cancer', 'Cell Line', 'Coumarin', 'Dose', 'Time', "Viability"]]

# Make the time feature numerical
newData['Time'] = newData['Time'].str.replace(r'\D', '', regex=True).astype(int);

# Concat the datas toghether
finalDataset = pd.concat([oldData, newData]);

finalDataset['Cancer'] = finalDataset['Cancer'].str.strip().str.lower().str.replace('\n', ' ', regex=False);

mapping = {
    # Prostate
    'prostate': 'prostate',
    'prostate cancer': 'prostate',

    # Breast
    'breast': 'breast',

    # Lung
    'lung': 'lung',
    'lung cancer': 'lung',
    'lung adenocarcinoma': 'lung',

    # Colon / Colorectal
    'colon': 'colon',
    'colorectal cancer': 'colon',
    'colon adenocarcinoma': 'colon',

    # Liver
    'liver': 'liver',
    'hepatocellular carcinoma': 'liver',

    # Cholangiocarcinoma
    'cholangiocarcinoma': 'cholangiocarcinoma',
    'intrahepatic cholangiocarcinoma': 'cholangiocarcinoma',

    # Cervical
    'cervical': 'cervical',
    'cervix': 'cervical',
    'cervical carcinoma': 'cervical',

    # Ovarian
    'ovarian': 'ovarian',
    'ovarian carcinoma': 'ovarian',
    'ovarian endometrioid adenocarcinoma': 'ovarian',

    # Endometrial / Uterine
    'uterine': 'endometrial',
    'endometrial': 'endometrial',

    # Renal
    'renal': 'renal',
    'renal cell carcinoma': 'renal',

    # Bone
    'bone': 'bone',
    'osteosarcoma': 'bone',

    # Skin
    'skin': 'skin',
    'melanoma': 'melanoma',

    # Brain tumors
    'glioma': 'glioma',
    'glioblastoma': 'glioma',
    'oligodendroglioma': 'glioma',
    'medulloblastoma': 'medulloblastoma',

    # Other organs
    'gastric': 'gastric',
    'pancreatic': 'pancreatic',
    'oral': 'oral',
    'salivary gland': 'salivary gland',
    'bladder': 'bladder',
    'nasopharyngeal': 'nasopharyngeal',
    'esophageal squamous cell carcinoma': 'esophageal',
    'gallbladder cancer': 'gallbladder',
    'leukemia/lymphoma': 'leukemia:lymphoma'
}

# Fix the Cancer names here
finalDataset['Cancer'] = finalDataset['Cancer'].replace(mapping);

# Final look at the Cancers in the dataset
print(finalDataset['Cancer'].unique())
print(finalDataset['Cancer'].value_counts())

# Strip whitespace from the columns names 
finalDataset.columns = finalDataset.columns.str.replace(" ", "", regex=False);
finalDataset.columns = finalDataset.columns.str.strip();

# Strip WhiteSpace from the coumarins as well
finalDataset['Coumarin'] = finalDataset['Coumarin'].str.strip().str.replace(" ", "", regex=False);

# Save the dataset
finalDataset.to_csv("Data/Final_Data.csv", index = False);