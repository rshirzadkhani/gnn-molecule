#%% Load Data
import pandas as pd
import numpy as np


data_path = "./data/raw/HIV.csv"
data = pd.read_csv(data_path)
data.head()

# %%
print(data.shape())
print(data["HIV_active"].value_counts())

#%%
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

mol_objects = data[4:31].values()