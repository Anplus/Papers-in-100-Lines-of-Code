import torch 
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

class iris_dataloader(Dataset):
    def __init__(self, data_path) -> None:
        super(Dataset, self).__init__()
        # path
        self.data_path = data_path
        #print(os.getcwd())
        assert os.path.exists(self.data_path), "dataset does not exist"

        df = pd.read_csv(self.data_path)
        # label mapping
        label_output = {'Iris-setosa':0, 'Iris-versicolor':1,'Iris-virginica':2}
        df['species'] = df['species'].map(label_output)
        data = df.iloc[:,0:4]
        label = df.iloc[:,4:5]
        # z value normalization
        data = (data-np.mean(data))/np.std(data)
        # dataForm to tensor
        self.data = torch.from_numpy(np.array(data, dtype='float32'))
        self.label = torch.from_numpy(np.array(label, dtype='float32'))
        self.data_num = len(label)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        self.data = list(self.data)
        self.label = list(self.label)
        return self.data[index], self.label[index]

# Test
# device setup
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # dataset division
# custom_dataset = iris_dataloader("Iris_Data.csv")
# train_size = int(len(custom_dataset)*0.7)
# val_size = int(len(custom_dataset)*0.2)
# test_size = len(custom_dataset)-train_size-val_size
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)