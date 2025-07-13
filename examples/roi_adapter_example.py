#!/usr/bin/env python3
"""
Example script demonstrating the ROI adapter functionality.

This script shows how to:
1. Use the ROI adapter to convert object detection to classification
2. Handle images with and without annotations
3. Use custom callback functions for ROI generation
4. Configure different ROI extraction parameters
"""

from pathlib import Path

from wildtrain.adapters.roi_adapter import ROIAdapter
from wildtrain.pipeline.loader import Loader

def custom_roi_callback(image_data,image_path=None):
    """
    Custom callback function for ROI generation.
    
    This function can use any model or algorithm to suggest ROIs.
    For example, you could use:
    - A pre-trained object detection model
    - Saliency detection
    - Edge detection
    - Random sampling with heuristics
    
    Args:
        image_path: Path to the image file
        image_data: OpenCV image data (numpy array)
        
    Returns:
        List of dictionaries with 'bbox' and 'class' keys
    """
    # This is a simple example that returns fixed ROIs
    # In practice, you would use a real model here
    
    height, width = image_data.shape[:2]
    
    # Example: suggest ROIs based on image size
    rois = []
    
    # Center ROI
    center_x = width // 2 - 50
    center_y = height // 2 - 50
    rois.append({
        "bbox": [center_x, center_y, 100, 100],
        "class": "center_object"
    })
    
    # Top-left ROI
    rois.append({
        "bbox": [50, 50, 80, 80],
        "class": "corner_object"
    })
    
    # Bottom-right ROI
    rois.append({
        "bbox": [width - 130, height - 130, 80, 80],
        "class": "corner_object"
    })
    
    return rois


def example_with_callback():
    """Example using custom callback for ROI generation."""
    print("\n=== ROI Adapter with Custom Callback ===")
    
    # Sample COCO data with no annotations
    coco_data = {
        "images": [
            {
                "id": 1,
                "file_name": "unannotated_image.jpg",
                "width": 640,
                "height": 480,
            }
        ],
        "annotations": [],  # No annotations
        "categories": [
            {"id": 1, "name": "object", "supercategory": "object"},
        ],
    }
    
    # Create ROI adapter with custom callback
    adapter = ROIAdapter(
        coco_data=coco_data,
        roi_callback=custom_roi_callback,
        random_roi_count=2,  # Additional random ROIs
        min_roi_size=48
    )
    
    # Convert to ROI format
    roi_data = adapter.convert()
    
    print(f"Generated {len(roi_data['roi_images'])} ROIs using callback")
    print(f"Statistics: {roi_data['statistics']}")
    
    # Print ROI information
    for i, roi_image in enumerate(roi_data["roi_images"]):
        roi_label = roi_data["roi_labels"][i]
        print(f"ROI {i+1}: {roi_label['class_name']} at {roi_image['bbox']}")
    
    return roi_data


def example_save_to_disk():
    """Example of saving ROI data to disk."""
    print("\n=== Saving ROI Data to Disk ===")
    
    ROOT = Path(r"D:\workspace\data\demo-dataset\framework_formats\roi")
    SOURCE_PATH = r"D:\workspace\savmap\coco\annotations\train.json"
        
    # Create output directory
    image_dir = ROOT / "images"
    image_dir.mkdir(exist_ok=True,parents=True)
    
    labels_dir = ROOT / "labels"
    labels_dir.mkdir(exist_ok=True,parents=True)  
    
    loader = Loader()
    split = "train"
    dataset_info, split_coco_data = loader.load(SOURCE_PATH, "coco", "roi-demo-savmap", bbox_tolerance=5, split_name=split)
    
    coco_data = split_coco_data[split]   
    adapter = ROIAdapter(coco_data=coco_data,
                         random_roi_count=1,
                         roi_box_size=128
                         )
    roi_data = adapter.convert()
    adapter.save(roi_data,output_images_dir=image_dir,output_labels_dir=labels_dir)
    
    print(f"Saved ROI data to {ROOT}")
    

if __name__ == "__main__":
    print("ROI Adapter Examples")
    print("=" * 50)
    
    try:
        # Run examples
        # example_basic_usage()
        # example_with_callback()
        example_save_to_disk()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc() 