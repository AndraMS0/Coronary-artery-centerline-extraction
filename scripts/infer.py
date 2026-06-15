import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from monai.networks.nets import UNet

# ===================================================
# DEVICE
# ===================================================
device = torch.device("cpu")

# ===================================================
# LOAD MODEL
# ===================================================
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ===================================================
# LOAD CT SCAN
# ===================================================
img = sitk.ReadImage(
    r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/data/0.nii.gz"
)

volume = sitk.GetArrayFromImage(img).astype(np.float32)

print("Original CT shape:", volume.shape)

# ===================================================
# PREPROCESSING (STANDARD CTA NORMALIZATION)
# ===================================================
volume = np.clip(volume, -200, 800)
volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

# ===================================================
# MEMORY SAFE DOWNSAMPLING
# ===================================================
volume = volume[::4, ::4, ::4]

print("Downsampled shape:", volume.shape)

# ===================================================
# CREATE TENSOR
# ===================================================
input_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()

print("Input tensor shape:", input_tensor.shape)

# ===================================================
# INFERENCE
# ===================================================
with torch.no_grad():
    pred = model(input_tensor)
    mask = (pred > 0.5).float()

print("Output shape:", mask.shape)

# ===================================================
# VISUALIZATION (THESIS IMPORTANT PART)
# ===================================================
slice_idx = volume.shape[0] // 2

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("CT Slice")
plt.imshow(volume[slice_idx], cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(mask[0, 0, slice_idx], cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# ===================================================
# SAVE OUTPUT MASK
# ===================================================
np.save("prediction_mask.npy", mask.numpy())

print("Mask saved as prediction_mask.npy")