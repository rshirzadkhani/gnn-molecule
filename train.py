import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from dataset import HIVDataset
from dataset_featurizer_2 import MoleculeDataset
from model import GNN
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Load datasets:
train_data = HIVDataset(root="data/", filename="HIV_train_oversampled.csv")
test_data = HIVDataset(root="data/", filename="HIV_test.csv")
print(train_data)
print(train_data[0].x.shape)
print(train_data[0].y.shape)
print(train_data[1].x.shape)
print(train_data[1].y.shape)


# Load model
model = GNN(input_dim=train_data[0].x.shape[1], embedding_size=1024, output_dim=2)
print(f"Number of parameters: {count_parameters(model)}")
# print(model)

# Define Loss & Optimizer
weights = torch.tensor([1, 10], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight = weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Batch Training
NUM_GRAPHS_PER_BATCH = 2
train_loader = DataLoader(train_data, 
                          batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
test_loader = DataLoader(test_data, 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train():
    total_loss = 0
    train_loss = []
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        print(batch)
        optimizer.zero_grad()
        y_pred = model(batch.x.float(),
                       batch.edge_attr.float(),
                       batch.edge_index,
                       batch.batch)
        
        loss = loss_fn(y_pred, batch.y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        total_loss += loss.item()
    print(total_loss)

if __name__ == "__main__":
    train()