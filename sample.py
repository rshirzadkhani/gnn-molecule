#%% Load Data
import pandas as pd
import numpy as np

data_path = "./data/raw/HIV.csv"
data = pd.read_csv(data_path)
data.head()

# %%
print(data.shape)
print(data["HIV_active"].value_counts())


# %% Check for Nan values
print(data.count())

# %%
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

samples = data["smiles"][0:4].values
sample_mol = [Chem.MolFromSmiles(smiles) for smiles in samples]
grid = Draw.MolsToGridImage(sample_mol, molsPerRow=4,
                            subImgSize=(200, 200))
grid
# %%
