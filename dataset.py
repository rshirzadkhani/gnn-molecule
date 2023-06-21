import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import numpy as np
import os

class HIVDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.filename = filename
        self.test = test
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
        i = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # if i == 0:
                mol_object = Chem.MolFromSmiles(mol["smiles"])
                node_feats = self._get_node_features(mol_object)
                edge_feats = self._get_edge_features(mol_object)
                adjacency_matrix = self._get_adjacency_matrix(mol_object)
                graph_labels = self._get_labels(self.data["HIV_active"])

                data = Data(x = node_feats,
                            edge_index=adjacency_matrix,
                            edge_attr=edge_feats,
                            y=graph_labels,
                            smiles=mol["smiles"])
                
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
                # i += 1

    def _get_node_features(self, mol):
        all_node_feats = []
        for atom in mol.GetAtoms():
            node_feats = []
            node_feats.append(atom.GetAtomicNum())
            node_feats.append(atom.GetDegree())
            node_feats.append(atom.GetFormalCharge())
            node_feats.append(atom.GetHybridization())
            node_feats.append(atom.GetIsAromatic())
            all_node_feats.append(node_feats)
        all_node_feats = np.asarray(all_node_feats)
        # print("node features", all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        all_edge_feats = []
        for edge in mol.GetBonds():
            edge_feats = []
            edge_feats.append(edge.GetBondTypeAsDouble())
            edge_feats.append(edge.IsInRing())
            all_edge_feats.append(edge_feats)
        all_edge_feats = np.asarray(all_edge_feats)
        # print("edge features", all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_matrix(self, mol):
        adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        # print(adj_matrix)
        row, col = np.where(adj_matrix)
        # print(row)
        # print(col)
        coo = np.array(list(zip(row, col)))
        # print(coo.shape)
        coo = np.reshape(coo, (2, -1))
        # print(coo.shape)
        return torch.tensor(coo, dtype=torch.long)
    
    def _get_labels(self, label):
        label = np.asarray(label)
        return torch.tensor(label, dtype=torch.int)


    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
if __name__ == "__main__":
    dataset = HIVDataset(root="./data", filename="HIV_train_oversampled.csv")