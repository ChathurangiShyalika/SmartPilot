import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.optim as optim
from sklearn.metrics import classification_report
import csv

# Import time series data
time_series_data=pd.read_csv('/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal-data/FF_Data_New.csv',low_memory=False)
time_series_data=time_series_data.rename(columns={'CycleState': 'cycle_state'})
time_series_data = time_series_data[['I_R04_Gripper_Load', 'I_R01_Gripper_Load', 'actual_state', 'cycle_state', 'Cam1', 'Cam2','Cycle_Count_New']]
time_series_data['actual_state'] = time_series_data['actual_state'].fillna(0).replace({
    'Normal': 0, 'NoNose': 1, 'NoBody1': 2, 'NoBody2': 3,
    'NoNose,NoBody2': 4, 'NoBody2,NoBody1': 5, 'NoNose,NoBody2,NoBody1': 6
})


noanomaly
nonose
NoNose,NoBody2
NoNose,NoBody2,NoBody1
NoBody1

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RocketDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.bboxes = []
        self._read_files()

    def _read_files(self):
        # Get list of all image files in root_dir
        image_files = os.listdir(self.root_dir)
        image_files_set = set(image_files)
        print(f"Total images in directory: {len(image_files)}")

        with open(self.csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                img_filename = os.path.basename(row['image_path'])
                if img_filename in image_files_set:
                    img_path = os.path.join(self.root_dir, img_filename)
                    xmin = float(row['xmin'])
                    ymin = float(row['ymin'])
                    xmax = float(row['xmax'])
                    ymax = float(row['ymax'])
                    self.image_paths.append(img_path)
                    self.bboxes.append((xmin, ymin, xmax, ymax))
                else:
                    print(f"Image file listed in CSV but not found in directory: {img_filename}")

        # Handle images in the directory not listed in the CSV
        for img_filename in image_files:
            if img_filename not in {os.path.basename(path) for path in self.image_paths}:
                print(f"Image file not listed in CSV: {img_filename}")
                self.image_paths.append(os.path.join(self.root_dir, img_filename))
                self.bboxes.append((0, 0, 224, 224))  # Default bbox

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        bbox = self.bboxes[idx]

        try:
            image = Image.open(img_path)
            image = image.crop(bbox)  # Crop to bounding box

            if self.transform:
                image = self.transform(image)

            return image

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None


def get_image_tensors1(root_dir1, csv_file1, transform=None):
    dataset = RocketDataset(root_dir1, csv_file1, transform=transform)
    image_tensors1 = [img for img in dataset if img is not None]
    print(f"Number of image tensors cam 1: {len(image_tensors1)}")
    return image_tensors1

def get_image_tensors2(root_dir2, csv_file2, transform=None):
    dataset = RocketDataset(root_dir2, csv_file2, transform=transform)
    image_tensors2 = [img for img in dataset if img is not None]
    print(f"Number of image tensors cam 2: {len(image_tensors2)}")
    return image_tensors2


# Usage
root_dir1 = '/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/cam1_combined'
csv_file1 = '/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/cam1_combined/cam1_combined_csv.csv'

root_dir2 = '/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/cam2_combined'
csv_file2 = '/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/cam2_combined/cam2_combined_csv.csv'
cam1_tensors = get_image_tensors1(root_dir1, csv_file1, transform=transform)
cam2_tensors = get_image_tensors2(root_dir2, csv_file2, transform=transform)


# Load the time series dataset
time_series_df=pd.read_csv('/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal-data/FF_Data_New.csv',low_memory=False)
time_series_image_paths1 = time_series_df['Cam1'].apply(os.path.basename).tolist()
image_folder_files1 = os.listdir(root_dir1)
matching_indices1 = [i for i, img_path in enumerate(time_series_image_paths1) if img_path in image_folder_files1]

time_series_image_paths2 = time_series_df['Cam2'].apply(os.path.basename).tolist()
image_folder_files2 = os.listdir(root_dir2)
matching_indices2 = [i for i, img_path in enumerate(time_series_image_paths2) if img_path in image_folder_files2]

combined_indices = list(set(matching_indices1) | set(matching_indices2))

combined_indices.sort()

len(combined_indices)

# Convert time series data to torch tensor
time_series_data_tensor = torch.tensor(time_series_data[['I_R04_Gripper_Load', 'I_R01_Gripper_Load', 'actual_state']].values, dtype=torch.float32)

# Generate next_sensor_values by shifting time_series_data by one position
next_sensor_values = torch.cat((time_series_data_tensor[1:], time_series_data_tensor[-1].unsqueeze(0)), dim=0)

time_series_data_tensor.shape

time_series_data_tensor


# Convert the list of tensors to a single tensor
cam1_tensors = torch.stack(cam1_tensors)
cam2_tensors = torch.stack(cam2_tensors)

# Concatenate the tensors along the batch dimension (0)
concat_tensor = torch.cat((cam1_tensors, cam2_tensors), dim=0)

print("Combined Tensor Shape:", concat_tensor.shape)
concat_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, time_series_data_tensor, next_sensor_values):
        self.time_series_data_tensor = time_series_data_tensor
        self.next_sensor_values = next_sensor_values

    def __len__(self):
        return len(self.time_series_data_tensor)

    def __getitem__(self, index):
        time_series_data_tensor = self.time_series_data_tensor[index]
        next_sensor = self.next_sensor_values[index]
        return time_series_data_tensor, next_sensor, index

# Create the dataset
dataset = CustomDataset(time_series_data_tensor, next_sensor_values)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

train_loader_count = sum(1 for _ in train_loader)
train_loader_count

image_data =concat_tensor
image_indices =combined_indices

image_data.shape


from efficientnet_pytorch import EfficientNet

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(TimeSeriesAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class FusionModel(nn.Module):
    def __init__(self, time_series_input_dim, image_output_dim, hidden_dim, output_dim):
        super(FusionModel, self).__init__()
        self.time_series_autoencoder = TimeSeriesAutoencoder(time_series_input_dim, hidden_dim, image_output_dim)
        self.image_encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = nn.Linear(image_output_dim + self.image_encoder._fc.in_features, output_dim)
        self.image_encoder._fc = nn.Identity()  # Remove the final classification layer

    def forward(self, time_series, image=None):
        reconstructed_time_series, latent_time_series = self.time_series_autoencoder(time_series)
        if image is not None:
            latent_image = self.image_encoder(image)
            latent_image = torch.flatten(latent_image, 1)  # Flatten the image features
            fused = torch.cat((latent_time_series, latent_image), dim=1)
            output = self.fc(fused)
            return output, reconstructed_time_series, latent_image
        else:
            return latent_time_series, reconstructed_time_series

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_series_input_dim = 3
image_output_dim = 128
hidden_dim = 64
output_dim = 3
model = FusionModel(time_series_input_dim, image_output_dim, hidden_dim, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (time_series, labels, index) in enumerate(train_loader):
        # Move time series data to the same device as the model
        time_series = time_series.to(device)
        labels = labels.to(device)

        # Check if the current index is in image_indices
        if index.item() in image_indices:
            #print(f"Index {index.item()} found in image_indices.")
            # Get the corresponding image from image_data
            image_idx = image_indices.index(index.item())
            image = image_data[image_idx].unsqueeze(0).to(device)  # Add batch dimension

            optimizer.zero_grad()
            outputs, reconstructed_time_series, latent_image = model(time_series, image)
            loss = criterion(outputs, labels) + criterion(reconstructed_time_series, time_series)

        else:
            optimizer.zero_grad()
            latent_time_series, reconstructed_time_series = model(time_series)
            loss = criterion(reconstructed_time_series, time_series)
            output=reconstructed_time_series


        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Validation loop with DataFrame creation
model.eval()
validation_loss = 0.0
output_list = []
original_list = []

with torch.no_grad():
    for i, (val_time_series, val_labels, val_index) in enumerate(val_loader):
        val_time_series = val_time_series.to(device)
        val_labels = val_labels.to(device)

        if val_index.item() in image_indices:
            # Get the corresponding image from image_data
            image_idx = image_indices.index(val_index.item())
            val_image = image_data[image_idx].unsqueeze(0).to(device)  # Add batch dimension

            val_outputs, val_reconstructed_time_series, val_latent_image = model(val_time_series, val_image)
            val_loss = criterion(val_outputs, val_labels) + criterion(val_reconstructed_time_series, val_time_series)
        else:
            val_latent_time_series, val_reconstructed_time_series = model(val_time_series)
            val_loss = criterion(val_reconstructed_time_series, val_time_series)
            val_outputs =val_reconstructed_time_series

        validation_loss += val_loss.item()

        # Store output and original time series data for DataFrame
        output_list.append(val_outputs.cpu().detach().numpy())
        original_list.append(val_labels.cpu().detach().numpy())

    average_val_loss = validation_loss / len(val_loader)
    print(f"Validation Loss: {average_val_loss:.4f}")

# Create DataFrame
df_validation = pd.DataFrame({
    'Output': output_list,
    'Original Time Series': original_list
})

# Extract the 3rd value from the 'outputs' and 'labels' columns into separate columns
df_validation['output_3rd_value'] = df_validation['Output'].apply(lambda x: x[0][2])  # Assuming 'outputs' is structured as a list of arrays
df_validation['label_3rd_value'] = df_validation['Original Time Series'].apply(lambda x: x[0][2])    # Assuming 'labels' is structured as a list of arrays

df_validation

from sklearn.metrics import classification_report

# Set negative values in "Predicted Output (3rd Value)" column to zero
df_validation['Predicted Output Label'] = df_validation['output_3rd_value'].apply(lambda x: max(x, 0))
df_validation['next'] = df_validation['label_3rd_value'].apply(lambda x: max(x, 0))


predicted_classes = df_validation['Predicted Output Label'].astype(int)
actual_classes = df_validation['next'].astype(int)


df_validation.to_csv('df_validation_100_epochs.csv', index=False)
report = classification_report(actual_classes, predicted_classes)
print(report)


torch.save(model.state_dict(), "n_decison_level_fusion_with pretrained_cnn_new1_chathu_100.pth")



