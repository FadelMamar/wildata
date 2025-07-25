from wildtrain.filters.filter_pipeline import FilterPipeline
from wildtrain.filters.algorithms import ClusteringFilter
from wildtrain.filters.filter_config import ClusteringFilterConfig
from wildtrain.pipeline.loader import Loader
from pathlib import Path
import logging

# Set up logging for demonstration
logging.basicConfig(level=logging.INFO)

# Path to your COCO annotation file
coco_json_path = r"D:\workspace\savmap\coco\annotations\train.json"  # <-- Update this path as needed

dataset_name = "my_coco_dataset"

# Loader setup (see loader.py for details)
print("Loading COCO data...")
loader = Loader()
dataset_info, split_data = loader.load(
    source_path=coco_json_path,
    source_format="coco",
    dataset_name=dataset_name,
    bbox_tolerance=5,
    split_name="train"
)
print("COCO data loaded.")

# Get the COCO data for the 'train' split
coco_data = split_data["train"]

if not coco_data['images']:
    print("WARNING: No images loaded from COCO data.")
else:
    print(f"First 5 images before filtering: {[img['file_name'] for img in coco_data['images'][:5]]}")
    print(f"First 5 image IDs before filtering: {[img['id'] for img in coco_data['images'][:5]]}")
    print(f"First 5 annotation image_ids before filtering: {[ann['image_id'] for ann in coco_data['annotations'][:5]]}")

# Configure the clustering filter
clustering_config = ClusteringFilterConfig(
    x_percent=0.3,  # Keep 30% of the data
)
print(f"Clustering filter config: {clustering_config}")

# Create the filter pipeline with a single clustering filter
clustering_filter = ClusteringFilter(config=clustering_config)
filter_pipeline = FilterPipeline(filters=[clustering_filter])

print("Applying filter pipeline...")
# Apply the filter pipeline
filtered_coco = filter_pipeline.filter(coco_data)
print("Filtering complete.")

# Print summary statistics
print(f"Original images: {len(coco_data['images'])}")
print(f"Filtered images: {len(filtered_coco['images'])}")
print(f"Original annotations: {len(coco_data['annotations'])}")
print(f"Filtered annotations: {len(filtered_coco['annotations'])}")

if filtered_coco['images']:
    print(f"First 5 images after filtering: {[img['file_name'] for img in filtered_coco['images'][:5]]}")
    print(f"First 5 image IDs after filtering: {[img['id'] for img in filtered_coco['images'][:5]]}")
    print(f"First 5 annotation image_ids after filtering: {[ann['image_id'] for ann in filtered_coco['annotations'][:5]]}")
else:
    print("WARNING: No images after filtering.")

# Log more details from the clustering filter if available
if hasattr(clustering_filter, 'CLUSTER_TRIALS'):
    print(f"Cluster trials: {clustering_filter.CLUSTER_TRIALS}")
if hasattr(clustering_filter, 'x_percent'):
    print(f"x_percent: {clustering_filter.x_percent}")
# If the filter stores silhouette scores, samples per cluster, or other stats, print them
if hasattr(clustering_filter, 'last_silhouette_scores'):
    print(f"Silhouette scores: {clustering_filter.last_silhouette_scores}")
if hasattr(clustering_filter, 'last_samples_per_cluster'):
    print(f"Samples per cluster: {clustering_filter.last_samples_per_cluster}") 