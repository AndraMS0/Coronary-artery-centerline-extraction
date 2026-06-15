import os
import numpy as np
import SimpleITK as sitk

prediction_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/predictions/"

files = sorted([f for f in os.listdir(prediction_dir) if f.endswith(".nii.gz")])

print("Predictions found:", len(files))
print("-" * 60)

for filename in files:
    path = os.path.join(prediction_dir, filename)

    img = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(img)

    vessel_voxels = np.sum(mask > 0)
    total_voxels = mask.size
    percentage = (vessel_voxels / total_voxels) * 100

    print(f"{filename}")
    print(f"  Shape: {mask.shape}")
    print(f"  Vessel voxels: {vessel_voxels}")
    print(f"  Percentage: {percentage:.6f}%")

    if vessel_voxels == 0:
        print("  Status: EMPTY MASK ❌")
    else:
        print("  Status: Contains predicted vessels ✅")

print("-" * 60)
print("Check completed.")