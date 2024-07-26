import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    accuracy_score, classification_report
)

import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RocketDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.bboxes = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(label_dir, img_file)
                        self.image_paths.append(img_path)
                        self.labels.append(int(label))

                        # Read corresponding CSV file for bounding box
                        csv_path = os.path.join(label_dir, "detections.csv")
                        bbox = self._read_bbox_from_csv(csv_path, img_file)
                        self.bboxes.append(bbox)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        bbox = self.bboxes[idx]

        image = Image.open(img_path)
        image = image.crop(bbox)  # Crop to bounding box

        if self.transform:
            image = self.transform(image)

        return image, label

    def _read_bbox_from_csv(self, csv_path, img_file):
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['image_path'].endswith(img_file):
                    xmin = int(float(row['xmin']))
                    ymin = int(float(row['ymin']))
                    xmax = int(float(row['xmax']))
                    ymax = int(float(row['ymax']))
                    return (xmin, ymin, xmax, ymax)
        # Return default bounding box if not found
        return (0, 0, 224, 224)  # Adjust as per your default bounding box strategy

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.n_patches, emb_size))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, n_patches, emb_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + n_patches, emb_size)
        x += self.pos_embed
        return x

class Attention(nn.Module):
    def __init__(self, emb_size, heads=8):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.proj = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, E // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (E // self.heads) ** -0.5
        att = att.softmax(dim=-1)
        x = (att @ v).transpose(1, 2).reshape(B, N, E)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, mlp_dim, dropout):
        super().__init__()
        self.attn = Attention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, heads=12, mlp_dim=3072, dropout=0.1, num_classes=5):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_size, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

# Load the dataset
data_dir = "/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/train_images"
dataset = RocketDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Compute class weights
labels = [label for _, label in dataset]
class_counts = pd.Series(labels).value_counts().sort_index()
class_weights = {i: 1.0 / count for i, count in class_counts.items()}
weights = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float32).to(device)

# Initialize the custom ViT model
model = VisionTransformer(num_classes=5)
model = model.to(device)

# Define loss function and optimizer with class weights
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

# Save the trained model
model_path = "custom_vit_model.pth"
torch.save(model.state_dict(), model_path)

# Load the test dataset
test_data_dir = "/work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/FF-multi-modal/val_images"
test_dataset = RocketDataset(test_data_dir, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model for evaluation
model = VisionTransformer(num_classes=5)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Evaluate the model
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

# Print classification report for detailed metrics
class_report = classification_report(
    all_labels, all_preds,
    target_names=["No Anomaly", "NoNose", "NoNose,NoBody2", "NoNose,NoBody2,NoBody1","NoBody1"])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

