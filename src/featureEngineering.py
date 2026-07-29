import numpy as np
import pandas as pd


# The first obvious feature to add will be the Molecular Weights (From PubChem)
MWs = {"Auraptene": 298.4,
       "Osthole": 244.28,
       "Esculetin": 178.14,
       "GalbanicAcid": 398.5,
       "Umbelliprenin": 366.5};

# The second Importand Feature will be LogP (XLogP From PubChem)
LogPs = {"Auraptene": 0,
         "Osthole": 0,
         "Esculetin": 0,
         "GalbanicAcid": 0,
         "Umbelliprenin": 7.1}

