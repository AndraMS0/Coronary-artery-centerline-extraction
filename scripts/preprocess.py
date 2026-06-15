import SimpleITK as sitk
import numpy as np

def load_ct(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)

    # CT normalization
    arr = arr.astype(np.float32)
    arr = np.clip(arr, -200, 800)
    arr = (arr - arr.min()) / (arr.max() - arr.min())

    # shape: [D, H, W]
    print("Loaded CT shape:", arr.shape)

    return arr