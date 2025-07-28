import os
import sys
from pathlib import Path
from wildata.adapters.utils import ExifGPSManager
from wildata.partitioning.utils import GPSUtils

def main():
    # Example usage: update EXIF GPS for all images in a folder using a CSV
    image_folder = r"D:\workspace\data\savmap_dataset_v2\images_splits"
    csv_path = "./path/to/your/gps_data.csv"  # Change to your CSV file
    output_dir = "./path/to/output"

    # Create the manager
    manager = ExifGPSManager()

    # Update all images in the folder using the CSV
    # The CSV must have columns: filename, latitude, longitude, [altitude]
    # Set inplace=True to modify images in place, or False to overwrite
    manager.update_folder_from_csv(
        image_folder=image_folder,
        csv_path=csv_path,
        output_dir=output_dir,
        skip_rows=0,
        filename_col="filename",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude"  # Optional, only if your CSV has altitude
    )
    print(f"EXIF GPS update complete for images in {image_folder} using {csv_path}")

def example_single_image_update():
    image_path = r"D:\workspace\data\savmap_dataset_v2\images_splits\00a033fefe644429a1e0fcffe88f8b39_1.JPG"
    output_path = "image_with_gps.jpg"
    manager = ExifGPSManager()
    manager.add_gps_to_image(
        input_path=image_path,
        output_path=output_path,
        latitude=40.689247,
        longitude=-74.044502,
        altitude=300.0
    )
    print("GPS image without gps: ", GPSUtils.get_gps_coord(image_path))
    print("GPS coordinates from image with gps: ", GPSUtils.get_gps_coord(output_path))


if __name__ == "__main__":
    # main() 
    example_single_image_update()