import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input is the 2d vector of the 2 features
        self.fc1 = nn.Linear(2, 16)
        # hidden layer
        self.fc2 = nn.Linear(16, 32)

        # output layer
        self.fc3 = nn.Linear(32, 1)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.dropout(self.elu(self.fc2(x)))
        x = self.fc3(x)
        return x

        
    
def train_nn(X_train, X_test, y_train, y_test, epochs=100, criterion = '', batch_size=32, lr=0.001, **kargs):

    
    
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # model.train()
        train_loss = 0.0
        for data, target in zip(X, y):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for i in range(len(X_test)):
                data, target = X_test[i].unsqueeze(0), Y_test[i].unsqueeze(0)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                predictions.append(output.item())
                targets.append(target.item())
        
        test_loss = test_loss / len(X_test)
        test_losses.append(test_loss)
        
        # 反标准化预测结果和目标值
        # predictions = sc_Y.inverse_transform(np.array(predictions).reshape(-1, 1))
        # targets = sc_Y.inverse_transform(np.array(targets).reshape(-1, 1))
        
        # 计算回归评估指标
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        # print(f'Epoch: {epoch}, Training Loss: {train_loss}, Testing Loss: {test_loss}, MSE: {mse}, MAE: {mae}')
    return model, train_losses, test_losses, mse, mae



