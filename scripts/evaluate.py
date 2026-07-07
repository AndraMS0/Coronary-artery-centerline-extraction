import os
import csv
import numpy as np
import SimpleITK as sitk

# ==================================================
# PATHS
# ==================================================
label_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/data/labels/"
prediction_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/predictions/"
results_dir = r"C:/Users/andra/OneDrive/Desktop/DIS/coronary_project/results/"

os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, "segmentation_metrics.csv")
txt_path = os.path.join(results_dir, "segmentation_summary.txt")

# ==================================================
# METRICS
# ==================================================
def compute_metrics(label, pred, eps=1e-8):
    label = (label > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    tp = np.sum((pred == 1) & (label == 1))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return dice, iou, precision, recall

# ==================================================
# EVALUATION
# ==================================================
label_files = sorted([
    f for f in os.listdir(label_dir)
    if f.endswith(".label.nii.gz")
])

results = []

print("Labels found:", len(label_files))

for label_file in label_files:
    case_id = label_file.replace(".label.nii.gz", "")

    label_path = os.path.join(label_dir, label_file)
    pred_path = os.path.join(prediction_dir, case_id + "_pred.nii.gz")

    if not os.path.exists(pred_path):
        print("Missing prediction for:", case_id)
        continue

    label_img = sitk.ReadImage(label_path)
    pred_img = sitk.ReadImage(pred_path)

    label = sitk.GetArrayFromImage(label_img)
    pred = sitk.GetArrayFromImage(pred_img)

    if label.shape != pred.shape:
        print(f"Shape mismatch for {case_id}: label {label.shape}, pred {pred.shape}")
        continue

    dice, iou, precision, recall = compute_metrics(label, pred)

    results.append({
        "case": case_id,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    })

    print(
        f"{case_id} | "
        f"Dice: {dice:.4f} | "
        f"IoU: {iou:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f}"
    )

# ==================================================
# SAVE CSV
# ==================================================
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "dice", "iou", "precision", "recall"])

    for r in results:
        writer.writerow([
            r["case"],
            r["dice"],
            r["iou"],
            r["precision"],
            r["recall"]
        ])

# ==================================================
# SAVE SUMMARY
# ==================================================
if len(results) > 0:
    mean_dice = np.mean([r["dice"] for r in results])
    mean_iou = np.mean([r["iou"] for r in results])
    mean_precision = np.mean([r["precision"] for r in results])
    mean_recall = np.mean([r["recall"] for r in results])

    with open(txt_path, "w") as f:
        f.write("Segmentation Evaluation Summary\n")
        f.write("===============================\n\n")
        f.write(f"Cases evaluated: {len(results)}\n")
        f.write(f"Mean Dice: {mean_dice:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Mean Precision: {mean_precision:.4f}\n")
        f.write(f"Mean Recall: {mean_recall:.4f}\n")

    print("\nAverage results:")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")

else:
    print("No cases evaluated.")

print("\nSaved CSV:", csv_path)
print("Saved summary:", txt_path)