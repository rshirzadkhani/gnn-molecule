import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import numpy as np
import os
import deepchem as dc 

class HIVDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        '''if this file found in raw directory download is not triggered'''
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        self.data.index = self.data["index"]
        featurizer = dc.feat.MolGraphConvfeaturizer(use_edge=True)
        i = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # if i == 0:
                f = featurizer.featurize(mol["smiles"])
                data = f.to_pyg_graph()
                data.y = self._get_label(mol["HIV_active"])
                data.smiles = mol["smiles"]
                if self.test:
                    torch.save(data, 
                               os.path.join(self.processed_dir, 
                                            f'data_test_{index}.pt'))
                else:
                    torch.save(data, 
                                os.path.join(self.processed_dir, 
                                            f'data_{index}.pt'))
                # i += 1
    
    def _get_labels(self, label):
        label = np.asarray(label)
        return torch.tensor(label, dtype=torch.int)


    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
    
if __name__ == "__main__":
    dataset = HIVDataset(root="./data")