import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the CSV data
csv_path = '/home/guoguo/projects/QML/data/gradient_results.csv'
data = pd.read_csv(csv_path)

# Prepare the input and output data
X = data[['B_freq', 'B_ampl', 'x_offset', 'y_offset', 'z_offset']].values
y = data[['g1_x', 'g1_y', 'g1_z', 'g2_x', 'g2_y', 'g2_z']].values

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network model
class GradientPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(GradientPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
output_size = y_train.shape[1]
model = GradientPredictor(input_size, output_size)

criterion = nn.MSELoss()
def custom_loss(output, target):
    g1_x_inv = 1.0 / (output[:, 0] + 1e-6)  # Adding a small value to prevent division by zero
    g2_y_inv = 1.0 / (output[:, 4] + 1e-6)
    return criterion(output, target) + g1_x_inv.sum() + g2_y_inv.sum() + output[:, 1].sum() + output[:, 2].sum() + output[:, 3].sum() + output[:, 5].sum()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(X_train_tensor)
    loss = custom_loss(predictions, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Find optimal input values for fixed gradients
fixed_gradients = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32)  # Example fixed gradients
X_optimized = X_train_tensor.clone().requires_grad_(True)
optimizer_opt = optim.Adam([X_optimized], lr=0.01)

for step in range(500):
    optimizer_opt.zero_grad()
    predicted_gradients = model(X_optimized)
    loss_opt = criterion(predicted_gradients, fixed_gradients)
    loss_opt.backward()
    optimizer_opt.step()

    if (step + 1) % 100 == 0:
        print(f'Step [{step + 1}/500], Optimization Loss: {loss_opt.item():.4f}')

# Denormalize optimized input values
optimal_input = scaler_X.inverse_transform(X_optimized.detach().numpy())
print("Optimal Input Values:", optimal_input)