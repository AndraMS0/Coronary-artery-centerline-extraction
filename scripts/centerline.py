import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# ==================================================
# PATHS
# ==================================================
prediction_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/predictions/"
output_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/centerlines/"

os.makedirs(output_dir, exist_ok=True)

# ==================================================
# MORPHOLOGICAL SKELETONIZATION
# ==================================================
def skeletonize_3d_scipy(mask):
    mask = mask.astype(bool)

    structure = ndimage.generate_binary_structure(3, 1)

    skeleton = np.zeros_like(mask, dtype=bool)
    eroded = mask.copy()

    while np.any(eroded):
        opened = ndimage.binary_opening(eroded, structure=structure)
        temp = eroded & ~opened
        skeleton |= temp
        eroded = ndimage.binary_erosion(eroded, structure=structure)

    return skeleton.astype(np.uint8)

# ==================================================
# PROCESS ALL PREDICTION MASKS
# ==================================================
files = sorted([
    f for f in os.listdir(prediction_dir)
    if f.endswith("_pred.nii.gz")
])

print("Prediction masks found:", len(files))

if len(files) == 0:
    raise RuntimeError("No prediction masks found. Run infer.py first.")

for filename in files:
    print("\nProcessing:", filename)

    mask_path = os.path.join(prediction_dir, filename)

    mask_img = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask_img)

    # binary mask
    mask = (mask > 0).astype(np.uint8)

    print("Mask shape:", mask.shape)
    print("Vessel voxels before cleanup:", int(np.sum(mask)))

    if np.sum(mask) == 0:
        print("Empty mask, skipping.")
        continue

    # cleanup
    mask = ndimage.binary_closing(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    mask = mask.astype(np.uint8)

    print("Vessel voxels after cleanup:", int(np.sum(mask)))

    # centerline extraction
    centerline = skeletonize_3d_scipy(mask)

    print("Centerline voxels:", int(np.sum(centerline)))

    output_name = filename.replace("_pred.nii.gz", "_centerline.nii.gz")
    output_path = os.path.join(output_dir, output_name)

    centerline_img = sitk.GetImageFromArray(centerline.astype(np.uint8))

    # keep metadata from prediction mask
    centerline_img.CopyInformation(mask_img)

    sitk.WriteImage(centerline_img, output_path)

    print("Saved:", output_path)

print("\nCenterline extraction completed.")