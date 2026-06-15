import os
import torch
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

# ==================================================
# PATHS
# ==================================================
image_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/data/images/"
label_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/data/labels/"

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ==================================================
# DATASET
# ==================================================
class CTDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))

        print("Images found:", len(self.images))
        print("Labels found:", len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.images[idx])
        lbl_path = os.path.join(self.label_dir, self.labels[idx])

        # ---------------- LOAD ----------------
        img = sitk.ReadImage(img_path)
        lbl = sitk.ReadImage(lbl_path)

        image = sitk.GetArrayFromImage(img).astype(np.float32)
        label = sitk.GetArrayFromImage(lbl).astype(np.float32)

        # ---------------- NORMALIZE CT ----------------
        image = np.clip(image, -200, 800)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # ---------------- BINARY MASK ----------------
        label = (label > 0).astype(np.float32)

        # ---------------- DOWNSAMPLE (VERY IMPORTANT) ----------------
        image = image[::2, ::2, ::2]
        label = label[::2, ::2, ::2]

        # ---------------- TO TENSOR ----------------
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)

        return image, label

# ==================================================
# DATA LOADER
# ==================================================
dataset = CTDataset(image_dir, label_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ==================================================
# MODEL (3D U-NET)
# ==================================================
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

# ==================================================
# LOSS + OPTIMIZER
# ==================================================
loss_function = DiceCELoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==================================================
# TRAINING LOOP
# ==================================================
epochs = 30

model.train()

for epoch in range(epochs):

    epoch_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")

# ==================================================
# SAVE MODEL
# ==================================================
torch.save(model.state_dict(), "model.pth")

print("Model saved successfully!")