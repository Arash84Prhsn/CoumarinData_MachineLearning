import numpy as np
import pandas as pd

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
ringCounts = {"Auraptene": 0,
              "Osthole": 0,
              "Esculetin": 0,
              "GalbanicAcid": 0,
              "Umbelliprenin": 0}

# FractionCsp3: The fraction of carbon atoms that are sp3 hybridized.
fCsp3s = {"Auraptene": 0.32,
          "Osthole": 0,
          "Esculetin": 0,
          "GalbanicAcid": 0,
          "Umbelliprenin": 35.5}