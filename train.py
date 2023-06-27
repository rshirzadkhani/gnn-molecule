import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from dataset import HIVDataset
from dataset_featurizer_2 import MoleculeDataset
from model import GNN
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, \
        precision_score, accuracy_score, f1_score
import mlflow.pytorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Load datasets:
train_data = HIVDataset(root="~/gnn-molecule/data/", filename="HIV_train.csv")
# test_data = HIVDataset(root="data/", filename="HIV_test.csv")
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Batch Training
NUM_GRAPHS_PER_BATCH = 2
train_loader = DataLoader(train_data[:50], 
                          batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
# test_loader = DataLoader(test_data, 
#                          batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def train(epoch):
    running_loss = 0
    all_preds = []
    all_labels = []
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch.x.float(),
                       batch.edge_index,
                       batch.edge_attr.float(),
                       batch.batch)
        
        loss = torch.sqrt(loss_fn(y_pred, batch.y))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
        all_preds.append(np.argmax(y_pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step

def test(epoch):
    total_loss = 0
    train_loss = []
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        y_pred = model(batch.x.float(),
                       batch.edge_index,
                       batch.edge_attr.float(),
                       batch.batch)
        
        loss = torch.sqrt(loss_fn(y_pred, batch.y))
        train_loss.append(loss.item())
        total_loss += loss.item()
    return loss    

def calculate_metrics(y_pred, y_true, epoch, type):
    print(y_pred, y_true)
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")    


with mlflow.start_run() as run:
    for epoch in range(10):
        model.train()
        loss = train(epoch)
        print(f'epoch {epoch} : Train Loss {loss}')
        mlflow.log_metric(key="Train-loss", value=float(loss), step=epoch)

        # Testing
        # if epoch % 5 == 0:
        #     model.eval()
        #     loss = test(epoch)
        #     print(f'epoch {epoch} : Test Loss {loss}')
        #     mlflow.log_metric(key="Test-loss", value=float(loss), step=epoch)