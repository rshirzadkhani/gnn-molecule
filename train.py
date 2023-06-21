import pandas as pd
from dataset_featurizer import HIVDataset
from dataset_featurizer_2 import MoleculeDataset
from model import GNN

# Load datasets:
# train_data = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")

train_data = HIVDataset(root="data/", filename="HIV_train_oversampled.csv")
test_data = HIVDataset("./data/raw/HIV_test.csv")

print(train_data.shape)
print(train_data[0].x.shape)
# Load model
model = GNN(embedding_size=train_data[0].x.shape[1])