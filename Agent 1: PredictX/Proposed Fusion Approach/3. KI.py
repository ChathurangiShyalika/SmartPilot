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
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

# Import time series data
#time_series_data=pd.read_csv('/content/drive/MyDrive/Event Prediction/Smart Manuf. Interns/Resources/FF_Data_New.csv')
time_series_data=pd.read_csv('/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal-data/FF_Data_New.csv',low_memory=False)

time_series_data=time_series_data.rename(columns={'CycleState': 'cycle_state'})
time_series_data = time_series_data[['I_R04_Gripper_Load', 'I_R01_Gripper_Load', 'actual_state', 'cycle_state', 'Cam1', 'Cam2','Cycle_Count_New']]
time_series_data['actual_state'] = time_series_data['actual_state'].fillna(0).replace({
    'Normal': 100, 'NoNose': 200, 'NoBody1': 300, 'NoBody2': 400,
    'NoNose,NoBody2': 500, 'NoBody2,NoBody1': 600, 'NoNose,NoBody2,NoBody1': 700
})


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
#time_series_df=pd.read_csv('/content/drive/MyDrive/Event Prediction/Smart Manuf. Interns/Resources/FF_Data_New.csv')
time_series_df=pd.read_csv('/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal-data/FF_Data_New.csv',low_memory=False)

time_series_image_paths1 = time_series_df['Cam1'].apply(os.path.basename).tolist()
image_folder_files = os.listdir(root_dir1)
matching_indices1 = [i for i, img_path in enumerate(time_series_image_paths1) if img_path in image_folder_files]

time_series_image_paths2 = time_series_df['Cam2'].apply(os.path.basename).tolist()
image_folder_files = os.listdir(root_dir2)
matching_indices2 = [i for i, img_path in enumerate(time_series_image_paths2) if img_path in image_folder_files]

combined_indices = list(set(matching_indices1) | set(matching_indices2))

combined_indices.sort()

len(combined_indices)

# Convert time series data to torch tensor
time_series_data_tensor = torch.tensor(time_series_data[['I_R04_Gripper_Load', 'I_R01_Gripper_Load', 'cycle_state','actual_state']].values, dtype=torch.float32)

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

#time_series_values = time_series_data_tensor.to(device)
#next_sensor_values = next_sensor_values.to(device)

class CustomDataset(Dataset):
    def __init__(self, time_series_data_tensor, next_sensor_values):
        self.time_series_data_tensor = time_series_data_tensor
        self.next_sensor_values = next_sensor_values

    def __len__(self):
        return len(self.time_series_data_tensor)

    def __getitem__(self, index):
        time_series_data_tensor = self.time_series_data_tensor[index]
        next_sensor = self.next_sensor_values[index]
        # Extracting cycle_state separately
        cycle_state = time_series_data_tensor[2]
        time_series_data_tensor_without_cycle_state = torch.cat((time_series_data_tensor[:2], time_series_data_tensor[3:]))
        next_sensor_data_tensor_without_cycle_state = torch.cat((next_sensor[:2], next_sensor[3:]))
        return time_series_data_tensor_without_cycle_state, next_sensor_data_tensor_without_cycle_state, cycle_state, index


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

time_series_data_tensor.shape

#image_indices

class_counts = time_series_data['actual_state'].value_counts().sort_index()
class_weights = {i: 1.0 / count for i, count in enumerate(class_counts)}
weights = np.array([class_weights[i] if i in class_weights else 0.0 for i in range(len(class_counts))])
weights /= weights.sum()
weights = torch.tensor(weights, dtype=torch.float32).to(device)

normal_ranges = {
    1: (0, 1000),
    2: (0, 1000),
    3: (0, 1000),
    4: (0, 1000),
    5: (0, 1000),
    6: (0, 1000),
    7: (0, 1000),
    8: (0, 1000),
    9: (0, 1000),
    10: (0, 1000),
    11: (0, 1000),
    12: (0, 1000),
    13: (0, 1000),
    14: (0, 1000),
    15: (0, 1000),
    16: (0, 1000),
    17: (0, 1000),
    18: (0, 1000),
    19: (0, 1000),
    20: (0, 1000),
    21: (0, 1000)}
    
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        targets = targets.long()
        third_values_targets = targets[:, 2]
        third_values_inputs = inputs[:, 2]
        class_to_index = {100: 0, 200: 1, 300:2, 400:3, 500:4, 600:5, 700:6}
        third_values_targets_int = [int(target.item()) for target in third_values_targets]
        mapped_targets = [class_to_index[target] for target in third_values_targets_int]
        sample_weights = self.weights[mapped_targets]
        weighted_loss = sample_weights * (inputs - targets) ** 2
        loss = weighted_loss.mean()
        return loss

criterion = WeightedMSELoss(weights)


class WeightedMSELossWithPenalty(nn.Module):
    def __init__(self, weights, normal_ranges):
        super(WeightedMSELossWithPenalty, self).__init__()
        self.weights = weights
        self.normal_ranges = normal_ranges

    def forward(self, inputs, targets):
        targets = targets.long()
        third_values_targets = targets[:, 2]
        third_values_inputs = inputs[:, 2]
        class_to_index = {100: 0, 200: 1, 300:2, 400:3, 500:4, 600:5, 700:6}
        third_values_targets_int = [int(target.item()) for target in third_values_targets]
        mapped_targets = [class_to_index[target] for target in third_values_targets_int]
        sample_weights = self.weights[mapped_targets]
        weighted_loss = sample_weights * (inputs - targets) ** 2
        weighted_loss = weighted_loss.mean(dim=1)
        loss = weighted_loss.mean()
        penalty = 0
        for i, time_series_sample in enumerate(time_series):
            sensor_value = targets[:, 1].item()  # Assuming 'I_RO1_Gripper_Load' is in the first column
            if cycle_state in self.normal_ranges:
                min_range, max_range = self.normal_ranges[cycle_state]
                if not (min_range <= sensor_value <= max_range):
                    penalty += (sensor_value - min_range)**2 if sensor_value < min_range else (sensor_value - max_range)**2

        loss = weighted_loss + penalty
        return loss.mean()
        
#only enocdr is frozen
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

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

class FusionModel(nn.Module):
    def __init__(self, time_series_input_dim, image_output_dim, hidden_dim, output_dim):
        super(FusionModel, self).__init__()
        self.time_series_autoencoder = TimeSeriesAutoencoder(time_series_input_dim, hidden_dim, image_output_dim)
        self.time_series_autoencoder.freeze_encoder()  # Freeze the encoder part
        self.image_encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = nn.Sequential(
            nn.Linear(image_output_dim + self.image_encoder._fc.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Adding Dropout for regularization
            nn.Linear(hidden_dim, output_dim)
        )
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
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)  # Only optimize trainable parameters


scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training loop

# Initialize the new loss function
criterion_with_penalty = WeightedMSELossWithPenalty(weights, normal_ranges)
criterion = WeightedMSELoss(weights)

# Training loop with early stopping, learning rate scheduler, and penalty
num_epochs = 50
patience = 10
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (time_series, labels, cycle_state, index) in enumerate(train_loader):
        # Move time series data to the same device as the model
        time_series = time_series.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if index.item() in image_indices:
            # Get the corresponding image from image_data
            image_idx = image_indices.index(index.item())
            image = image_data[image_idx].unsqueeze(0).to(device)  # Add batch dimension

            outputs, reconstructed_time_series, latent_image = model(time_series, image)
            loss_outputs = criterion(outputs, labels)
            loss_reconstructed = criterion(reconstructed_time_series, labels)
            loss = loss_outputs + loss_reconstructed
        else:
            latent_time_series, reconstructed_time_series = model(time_series)
            loss_reconstructed = criterion(reconstructed_time_series, labels)
            loss = loss_reconstructed

        # Use the criterion with penalty starting from the third epoch
        if epoch >= 3:
            #loss = criterion_with_penalty(outputs, labels)
            if index.item() in image_indices:
              # Get the corresponding image from image_data
              image_idx = image_indices.index(index.item())
              image = image_data[image_idx].unsqueeze(0).to(device)  # Add batch dimension

              outputs, reconstructed_time_series, latent_image = model(time_series, image)
              loss_outputs = criterion_with_penalty(outputs, labels)
              loss_reconstructed = criterion_with_penalty(reconstructed_time_series, labels)
              loss = loss_outputs + loss_reconstructed
            else:
              latent_time_series, reconstructed_time_series = model(time_series)
              loss_reconstructed = criterion_with_penalty(reconstructed_time_series, labels)
              loss = loss_reconstructed

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Validation step
    model.eval()
    val_loss = 0.0

    # Temporary lists to store current epoch validation data
    current_original_time_series_list = []
    current_labels_list = []
    current_outputs_list = []
    cycle_states_list = []

    with torch.no_grad():
        for i, (time_series, labels, cycle_state, index) in enumerate(val_loader):
            # Move data to device
            time_series = time_series.to(device)
            labels = labels.to(device)

            if index.item() in image_indices:
                # Get the corresponding image from image_data
                image_idx = image_indices.index(index.item())
                image = image_data[image_idx].unsqueeze(0).to(device)  # Add batch dimension

                outputs, reconstructed_time_series, latent_image = model(time_series, image)
            else:
                latent_time_series, reconstructed_time_series = model(time_series)
                outputs = reconstructed_time_series

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Append data to current epoch lists
            current_original_time_series_list.append(time_series.cpu().numpy())
            current_labels_list.append(labels.cpu().numpy())
            current_outputs_list.append(outputs.cpu().numpy())
            cycle_states_list.append(cycle_state.cpu().numpy())

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Scheduler step
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        
        # Save the current epoch validation data as final results
        final_original_time_series_list = current_original_time_series_list
        final_labels_list = current_labels_list
        final_outputs_list = current_outputs_list
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping")
        break

# Create DataFrame after all epochs
df = pd.DataFrame({
    'Original Time Series': final_original_time_series_list,
    'Labels': final_labels_list,
    'Outputs': final_outputs_list,
    'Cycle States': cycle_states_list
})
print(df)


# Extract the 3rd value from the 'outputs' and 'labels' columns into separate columns
df['output_3rd_value'] = df['Outputs'].apply(lambda x: x[0][2])  # Assuming 'outputs' is structured as a list of arrays
df['label_3rd_value'] = df['Original Time Series'].apply(lambda x: x[0][2])    # Assuming 'labels' is structured as a list of arrays

from sklearn.metrics import classification_report

# Set negative values in "Predicted Output (3rd Value)" column to zero
df['Predicted Output Label'] = df['output_3rd_value'].apply(lambda x: max(x, 0))
df['next'] = df['label_3rd_value'].apply(lambda x: max(x, 0))

predicted_classes = df['Predicted Output Label'].astype(int)
actual_classes = df['next'].astype(int)

report = classification_report(actual_classes, predicted_classes)
print(report)

df.to_csv('df_50_epochs_frozen_encoder_weighted_loss_optimized_KI.csv', index=False)

torch.save(model.state_dict(), "decison_level_fusion_with pretrained_cnn_new_frozen_encoder_weighted_loss_epoch_50_optimized_KI.pth")


