import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from monai.networks.nets import UNet

# ===================================================
# PATHS
# ===================================================
image_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/data/images/"
output_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/predictions/"
model_path = "model.pth"

os.makedirs(output_dir, exist_ok=True)

# ===================================================
# DEVICE
# ===================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

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

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===================================================
# INFERENCE FOR ALL SCANS
# ===================================================
files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])

print("Files found:", len(files))

for filename in files:

    print("\nProcessing:", filename)

    ct_path = os.path.join(image_dir, filename)

    # ---------------- LOAD CT ----------------
    img = sitk.ReadImage(ct_path)
    volume = sitk.GetArrayFromImage(img).astype(np.float32)

    print("Original shape:", volume.shape)

    # ---------------- PREPROCESS ----------------
    volume = np.clip(volume, -200, 800)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # MUST match training downsampling
    volume = volume[::2, ::2, ::2]

    print("Downsampled shape:", volume.shape)

    # ---------------- TENSOR ----------------
    input_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(device)

    # ---------------- PREDICT ----------------
    with torch.no_grad():
        pred = model(input_tensor)
        prob = torch.sigmoid(pred)
        mask = (prob > 0.5).float()

    mask_np = mask[0, 0].cpu().numpy().astype(np.uint8)

    print("Output mask shape:", mask_np.shape)

    # ---------------- SAVE NIFTI MASK ----------------
    output_name = filename.replace(".nii.gz", "_pred.nii.gz")
    output_path = os.path.join(output_dir, output_name)

    mask_img = sitk.GetImageFromArray(mask_np)
    sitk.WriteImage(mask_img, output_path)

    print("Saved:", output_path)

# ===================================================
# OPTIONAL VISUALIZATION OF LAST CASE
# ===================================================
slice_idx = volume.shape[0] // 2

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("CT Slice")
plt.imshow(volume[slice_idx], cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(mask_np[slice_idx], cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nInference completed for all scans.")