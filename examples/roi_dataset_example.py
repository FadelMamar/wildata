# If you don't have matplotlib, install it with: pip install matplotlib
from pathlib import Path

from wildata.datasets.roi import ROIDataset, load_all_roi_datasets
from tqdm import tqdm
# Set the root data directory and dataset name
root_data_directory = Path(r"D:\workspace\data\demo-dataset")
dataset_name = "savmap"
split = "train"

# Instantiate the dataset
roi_dataset = ROIDataset(
    dataset_name=dataset_name,
    split=split,
    root_data_directory=root_data_directory,
)

for _ in tqdm(roi_dataset):
    continue

print(f"Number of samples in {split} split: {len(roi_dataset)}")

# Show the first image and label
image, label = roi_dataset[0]
print(f"Label: {label} (class name: {roi_dataset.class_mapping[label.item()]})")
print(roi_dataset.class_mapping)
