import torch
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from dataloader import iris_dataloader
# model
class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
# device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset division
custom_dataset = iris_dataloader("./data/Iris_Data.csv")
train_size = int(len(custom_dataset)*0.7)
val_size = int(len(custom_dataset)*0.2)
test_size = len(custom_dataset)-train_size-val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle= False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle= False)
#
print('Dataset Size', len(train_loader))

# inference
def infer(model, dataset, device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas, label = data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(predict_y, label.to(device)).sum().item()
    acc = acc_num/len(dataset)
    return acc

def main(learning_rate = 0.005, epochs = 20):
    # model and train
    model = NN(4, 12, 3).to(device)
    loss_f = nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=learning_rate)
    # save weight
    save_path = os.path.join(os.getcwd(),'result/weight')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # training
    for epoch in range(epochs):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0
        train_bar = tqdm(train_loader,file=sys.stdout, ncols=100)
        for datas in train_bar:
            # initialization
            data, label = datas
            label = label.squeeze(-1)
            sample_num += data.shape[0]
            optimizer.zero_grad()
            # output
            outputs = model(data.to(device))
            pred_class = torch.max(outputs, dim=1)[1] # return a tuple()
            acc_num = torch.eq(pred_class, label.to(device)).sum()
            # loss computation
            loss = loss_f(outputs, label.to(device, dtype=torch.long))
            loss.backward()
            optimizer.step()
            train_acc = acc_num/sample_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, loss, train_acc)
        val_acc = infer(model, val_loader, device)
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss, train_acc)
        torch.save(model.state_dict(), os.path.join(save_path, 'nn.pth'))

        train_acc = 0
        val_acc = 0

    print("finished Training")
    test_acc = infer(model, test_loader, device)
    print("test accuracy", test_acc)

if __name__ == "__main__":
    main()
