import os
import random
import csv
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
# CONFIG
# ==================================================
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-4
VAL_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr


def normalize_ct(volume):
    volume = np.clip(volume, -200, 800)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume.astype(np.float32)


def pad_if_needed(image, label, patch_size):
    d, h, w = image.shape
    pd, ph, pw = patch_size

    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant"
        )

        label = np.pad(
            label,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant"
        )

    return image, label


def crop_patch(image, label, patch_size):
    image, label = pad_if_needed(image, label, patch_size)

    d, h, w = image.shape
    pd, ph, pw = patch_size

    vessel_voxels = np.argwhere(label > 0)

    # 70% chance to crop around vessel voxels
    if len(vessel_voxels) > 0 and random.random() < 0.7:
        cz, cy, cx = vessel_voxels[random.randint(0, len(vessel_voxels) - 1)]
    else:
        cz = random.randint(0, d - 1)
        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)

    z1 = max(0, min(cz - pd // 2, d - pd))
    y1 = max(0, min(cy - ph // 2, h - ph))
    x1 = max(0, min(cx - pw // 2, w - pw))

    image_patch = image[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw]
    label_patch = label[z1:z1 + pd, y1:y1 + ph, x1:x1 + pw]

    return image_patch, label_patch


def dice_score(pred, target, eps=1e-8):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2.0 * intersection + eps) / (union + eps)

# ==================================================
# DATASET
# ==================================================
class CoronaryDataset(Dataset):
    def __init__(self, pairs, image_dir, label_dir, patch_size):
        self.pairs = pairs
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_file, label_file = self.pairs[idx]

        img_path = os.path.join(self.image_dir, img_file)
        lbl_path = os.path.join(self.label_dir, label_file)

        image = load_nifti(img_path)
        label = load_nifti(lbl_path)

        image = normalize_ct(image)
        label = (label > 0).astype(np.float32)

        image_patch, label_patch = crop_patch(image, label, self.patch_size)

        image_tensor = torch.from_numpy(image_patch).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label_patch).unsqueeze(0).float()

        return image_tensor, label_tensor

# ==================================================
# IMAGE-LABEL PAIRING
# ==================================================
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.endswith(".img.nii.gz")
])

pairs = []

for img_file in image_files:
    case_id = img_file.replace(".img.nii.gz", "")
    label_file = case_id + ".label.nii.gz"

    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, label_file)

    if os.path.exists(lbl_path):
        pairs.append((img_file, label_file))
    else:
        print("Missing label for:", img_file)

print("Total matched pairs:", len(pairs))

if len(pairs) == 0:
    raise RuntimeError("No image-label pairs found. Check file names and paths.")

# ==================================================
# TRAIN / VALIDATION SPLIT
# ==================================================
random.seed(42)
random.shuffle(pairs)

val_size = int(len(pairs) * VAL_SPLIT)

val_pairs = pairs[:val_size]
train_pairs = pairs[val_size:]

print("Training pairs:", len(train_pairs))
print("Validation pairs:", len(val_pairs))

# ==================================================
# DATA LOADERS
# ==================================================
train_dataset = CoronaryDataset(
    train_pairs,
    image_dir,
    label_dir,
    PATCH_SIZE
)

val_dataset = CoronaryDataset(
    val_pairs,
    image_dir,
    label_dir,
    PATCH_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)

# ==================================================
# MODEL
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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ==================================================
# TRAINING
# ==================================================
best_val_dice = 0.0

with open("training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_function(pred, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = loss_function(pred, y)
            dice = dice_score(pred, y)

            val_loss += loss.item()
            val_dice += dice.item()

    val_loss = val_loss / len(val_loader)
    val_dice = val_dice / len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    with open("training_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            val_dice
        ])

    # ---------------- SAVE BEST MODEL ----------------
    if val_dice > best_val_dice:
        best_val_dice = val_dice

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved!")

# ==================================================
# SAVE FINAL MODEL
# ==================================================
torch.save(
    model.state_dict(),
    "model.pth"
)

print("Training completed.")
print("Best validation Dice:", best_val_dice)
print("Final model saved as model.pth")
print("Best model saved as best_model.pth")