import os
import torch
import numpy as np
import SimpleITK as sitk

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# ===================================================
# PATHS
# ===================================================
image_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/data/images/"
output_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/predictions/"
model_path = "best_model.pth"

os.makedirs(output_dir, exist_ok=True)

# ===================================================
# CONFIG
# ===================================================
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 1
THRESHOLD = 0.5

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

print("Loaded model:", model_path)

# ===================================================
# PREPROCESS FUNCTION
# ===================================================
def preprocess_ct(volume):
    volume = volume.astype(np.float32)
    volume = np.clip(volume, -200, 800)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume

# ===================================================
# INFERENCE FOR ALL SCANS
# ===================================================
files = sorted([
    f for f in os.listdir(image_dir)
    if f.endswith(".img.nii.gz")
])

print("Files found:", len(files))

for filename in files:

    print("\nProcessing:", filename)

    ct_path = os.path.join(image_dir, filename)

    # ---------------- LOAD CT ----------------
    img = sitk.ReadImage(ct_path)
    volume = sitk.GetArrayFromImage(img)

    print("Original shape:", volume.shape)

    # ---------------- PREPROCESS ----------------
    volume = preprocess_ct(volume)

    # ---------------- TENSOR ----------------
    input_tensor = (
        torch.from_numpy(volume)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    print("Input tensor shape:", input_tensor.shape)

    # ---------------- SLIDING WINDOW INFERENCE ----------------
    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=input_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=0.25
        )

        prob = torch.sigmoid(pred)
        mask = (prob > THRESHOLD).float()

    mask_np = mask[0, 0].cpu().numpy().astype(np.uint8)

    print("Output mask shape:", mask_np.shape)

    # ---------------- SAVE NIFTI MASK ----------------
    case_id = filename.replace(".img.nii.gz", "")
    output_name = case_id + "_pred.nii.gz"
    output_path = os.path.join(output_dir, output_name)

    mask_img = sitk.GetImageFromArray(mask_np)

    # keep spacing, origin, direction from original CT
    mask_img.CopyInformation(img)

    sitk.WriteImage(mask_img, output_path)

    print("Saved:", output_path)

print("\nInference completed for all scans.")