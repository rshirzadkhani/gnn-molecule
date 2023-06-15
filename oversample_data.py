import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("./data/raw/HIV.csv")
data_train, data_test = train_test_split(data, test_size=0.5)

data_train.to_csv("./data/raw/HIV_train.csv")
data_test.to_csv("./data/raw/HIV_test.csv")


# data = pd.read_csv("./data/raw/HIV_train.csv")
class_1 = data_train["HIV_active"].value_counts()[0]
class_2 = data_train["HIV_active"].value_counts()[1]
multiplier = int(np.round(class_1 / class_2))
print(multiplier)
replicated_pos = [data_train[data_train["HIV_active"] == 1]] * multiplier
data_train = data_train.append(replicated_pos, ignore_index=True)
print(data_train.shape)

data_train = data_train.sample(frac=1).reset_index(drop=True)

index = range(0, data_train.shape[0])
data_train.index = index
data_train["index"] = data_train.index
print(data.head())

data_train.to_csv("./data/raw/HIV_train_oversampled.csv")