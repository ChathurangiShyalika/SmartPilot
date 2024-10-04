#autoencoder to compare with next values
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

# Load and preprocess the data
time_series_data = pd.read_csv('/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal-data/FF_Data_New.csv', low_memory=False)
time_series_data = time_series_data[['I_R04_Gripper_Load', 'I_R01_Gripper_Load', 'actual_state']]
time_series_data['actual_state'].fillna(0, inplace=True)
time_series_data['actual_state'] = time_series_data['actual_state'].replace({
    'Normal': 0, 'NoNose': 1, 'NoBody1': 2, 'NoBody2': 3, 'NoNose,NoBody2': 4,
    'NoBody2,NoBody1': 5, 'NoNose,NoBody2,NoBody1': 6
})

# Convert the DataFrame to a PyTorch tensor
data_tensor = torch.tensor(time_series_data.values, dtype=torch.float32)

# Calculate next time series values
next_sensor_values = torch.cat((data_tensor[1:], data_tensor[-1].unsqueeze(0)), dim=0)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, time_series_data, next_sensor_values):
        self.time_series_data = time_series_data
        self.next_sensor_values = next_sensor_values

    def __len__(self):
        return len(self.time_series_data)

    def __getitem__(self, index):
        time_series = self.time_series_data[index]
        next_sensor = self.next_sensor_values[index]

        return time_series, next_sensor


# Create the dataset
dataset = CustomDataset(data_tensor, next_sensor_values)

# Split the dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)


# Define the Autoencoder class
class AutoencoderRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoencoderRegressor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize the model, loss function, and optimizer
input_size = data_tensor.shape[1]
hidden_size = 10  # Adjust as needed
model = AutoencoderRegressor(input_size, hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for idx, (time_series_data, next_sensor_values) in enumerate(train_loader):
        time_series_data, next_sensor_values = time_series_data.to(device), next_sensor_values.to(device)

        optimizer.zero_grad()
        outputs = model(time_series_data)
        loss = criterion(outputs, next_sensor_values)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for time_series_data, labels in val_loader:
            time_series_data, labels = time_series_data.to(device), labels.to(device)

            outputs = model(time_series_data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

val_data_list = []
predicted_outputs_list = []

# Set model to evaluation mode
model.eval()

# Loop through validation loader
with torch.no_grad():
    for time_series_data, labels in val_loader:
        time_series_data, labels = time_series_data.to(device), labels.to(device)

        outputs = model(time_series_data)
        #take next timestamp value and compare
        val_data_list.append(labels.cpu().numpy().flatten())
        predicted_outputs_list.append(outputs.cpu().numpy().flatten())

# Convert lists to NumPy arrays
val_data_array = np.array(val_data_list)
predicted_outputs_array = np.array(predicted_outputs_list)

# Create DataFrame
df = pd.DataFrame({
    'Sensor_1': val_data_array[:, 0],
    'Sensor_2': val_data_array[:, 1],
    'Actual_Label': val_data_array[:, 2],
    'Predicted_sensor 1': predicted_outputs_array[:, 0],  
    'Predicted_sensor 2': predicted_outputs_array[:, 1],  
    'Predicted_Label': predicted_outputs_array[:, 2]  
})

# Display the DataFrame
df

from sklearn.metrics import classification_report

df['Predicted_Label'] = df['Predicted_Label'].apply(lambda x: max(x, 0))

predicted_classes = df['Predicted_Label'].astype(int)
actual_classes = df['Actual_Label']

report = classification_report(actual_classes, predicted_classes)
print(report)

df.to_csv('autoencoder.csv', index=False)

torch.save(model.state_dict(), "autoencoder_50_epoch_new.pth")


