"""
Streamlit UI for WildData CLI functionalities.
"""

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import streamlit as st
import yaml

# Add the src directory to Python path to import wildata modules
from wildata.cli.models import (
    BulkCreateROIDatasetConfig,
    BulkImportDatasetConfig,
    ExifGPSUpdateConfig,
    ImportDatasetConfig,
    ROIDatasetConfig,
)
from wildata.config import ROOT
from wildata.pipeline import DataPipeline


def run_cli_command(command: str, args: list) -> tuple[bool, str]:
    """Run a CLI command and return success status and output."""
    try:
        result = subprocess.Popen(
            ["uv", "run", "wildata"] + [command] + args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=os.environ.copy(),
        )
        stdout, stderr = result.communicate()
        return result.returncode == 0, stdout + stderr
    except Exception as e:
        return False, f"Error running command: {str(e)}"


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to temporary location and return path."""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None


def create_config_from_ui(config_class, **kwargs):
    """Create a config object from UI inputs."""
    try:
        return config_class(**kwargs)
    except Exception as e:
        st.error(f"Configuration error: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="WildData Pipeline",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🦁 WildData Pipeline")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📥 Import & Export",
            "🎯 ROI Operations",
            "📍 GPS Operations",
            "👁️ Visualization",
            "📊 Dataset Management",
        ]
    )

    # Tab 1: Import & Export
    with tab1:
        st.header("📥 Import & Export")

        # Import Dataset
        st.subheader("Import Dataset")
        with st.expander("Import Dataset", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Using YAML Config File**")
                config_path = st.text_input("Config File Path", key="import_config")

                if config_path and st.button("Import Dataset (Config)"):
                    success, output = run_cli_command(
                        "import-dataset", ["--config", config_path.strip(), "--verbose"]
                    )
                    if success:
                        st.success("✅ Dataset imported successfully!")
                    else:
                        st.error(f"❌ Import failed: {output}")

            with col2:
                st.markdown("**Using Command Line Arguments**")
                source_path = st.text_input("Source Path", key="import_source")
                source_format = st.selectbox(
                    "Source Format", ["coco", "yolo", "ls"], key="import_format"
                )
                dataset_name = st.text_input("Dataset Name", key="import_name")

                if st.button("Import Dataset (CLI)"):
                    if source_path and dataset_name:
                        args = [
                            source_path,
                            "--format",
                            source_format,
                            "--name",
                            dataset_name,
                            "--verbose",
                        ]
                        success, output = run_cli_command("import-dataset", args)
                        if success:
                            st.success("✅ Dataset imported successfully!")
                        else:
                            st.error(f"❌ Import failed: {output}")
                    else:
                        st.error("Please provide source path and dataset name")

        # Bulk Import Datasets
        st.subheader("Bulk Import Datasets")
        with st.expander("Bulk Import Datasets", expanded=True):
            uploaded_bulk_config = st.file_uploader(
                "Upload YAML config file for bulk import",
                type=["yaml", "yml"],
                key="bulk_import_config",
            )

            if uploaded_bulk_config:
                config_path = save_uploaded_file(uploaded_bulk_config)
                if config_path and st.button("Bulk Import Datasets"):
                    success, output = run_cli_command(
                        "bulk-import-datasets", ["--config", config_path, "--verbose"]
                    )
                    if success:
                        st.success("✅ Bulk import completed successfully!")
                    else:
                        st.error(f"❌ Bulk import failed: {output}")
                    os.unlink(config_path)

    # Tab 2: ROI Operations
    with tab2:
        st.header("🎯 ROI Operations")

        # Create ROI Dataset
        st.subheader("Create ROI Dataset")
        with st.expander("Create ROI Dataset", expanded=True):
            uploaded_roi_config = st.file_uploader(
                "Upload YAML config file for ROI creation",
                type=["yaml", "yml"],
                key="roi_config",
            )

            if uploaded_roi_config:
                config_path = save_uploaded_file(uploaded_roi_config)
                if config_path and st.button("Create ROI Dataset"):
                    success, output = run_cli_command(
                        "create-roi-dataset", ["--config", config_path, "--verbose"]
                    )
                    if success:
                        st.success("✅ ROI dataset created successfully!")
                    else:
                        st.error(f"❌ ROI creation failed: {output}")
                    os.unlink(config_path)

        # Bulk Create ROI Datasets
        st.subheader("Bulk Create ROI Datasets")
        with st.expander("Bulk Create ROI Datasets", expanded=True):
            uploaded_bulk_roi_config = st.file_uploader(
                "Upload YAML config file for bulk ROI creation",
                type=["yaml", "yml"],
                key="bulk_roi_config",
            )

            if uploaded_bulk_roi_config:
                config_path = save_uploaded_file(uploaded_bulk_roi_config)
                if config_path and st.button("Bulk Create ROI Datasets"):
                    success, output = run_cli_command(
                        "bulk-create-roi-datasets",
                        ["--config", config_path, "--verbose"],
                    )
                    if success:
                        st.success("✅ Bulk ROI creation completed successfully!")
                    else:
                        st.error(f"❌ Bulk ROI creation failed: {output}")
                    os.unlink(config_path)

    # Tab 3: GPS Operations
    with tab3:
        st.header("📍 GPS Operations")

        st.subheader("Update GPS from CSV")
        with st.expander("Update GPS from CSV", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Using YAML Config File**")
                uploaded_gps_config = st.file_uploader(
                    "Upload YAML config file for GPS update",
                    type=["yaml", "yml"],
                    key="gps_config",
                )

                if uploaded_gps_config:
                    config_path = save_uploaded_file(uploaded_gps_config)
                    if config_path and st.button("Update GPS (Config)"):
                        success, output = run_cli_command(
                            "update-gps-from-csv",
                            ["--config", config_path, "--verbose"],
                        )
                        if success:
                            st.success("✅ GPS data updated successfully!")
                        else:
                            st.error(f"❌ GPS update failed: {output}")
                        os.unlink(config_path)

            with col2:
                st.markdown("**Using Command Line Arguments**")
                image_folder = st.text_input(
                    "Image Folder Path", key="gps_image_folder"
                )
                csv_path = st.text_input("CSV File Path", key="gps_csv_path")
                output_dir = st.text_input("Output Directory", key="gps_output_dir")

                # Optional parameters
                skip_rows = st.number_input(
                    "Skip Rows", min_value=0, value=0, key="gps_skip_rows"
                )
                filename_col = st.text_input(
                    "Filename Column", value="filename", key="gps_filename_col"
                )
                lat_col = st.text_input(
                    "Latitude Column", value="latitude", key="gps_lat_col"
                )
                lon_col = st.text_input(
                    "Longitude Column", value="longitude", key="gps_lon_col"
                )
                alt_col = st.text_input(
                    "Altitude Column", value="altitude", key="gps_alt_col"
                )

                if st.button("Update GPS (CLI)"):
                    if image_folder and csv_path and output_dir:
                        args = [
                            "--image-folder",
                            image_folder,
                            "--csv",
                            csv_path,
                            "--output",
                            output_dir,
                            "--skip-rows",
                            str(skip_rows),
                            "--filename-col",
                            filename_col,
                            "--lat-col",
                            lat_col,
                            "--lon-col",
                            lon_col,
                            "--alt-col",
                            alt_col,
                            "--verbose",
                        ]
                        success, output = run_cli_command("update-gps-from-csv", args)
                        if success:
                            st.success("✅ GPS data updated successfully!")
                        else:
                            st.error(f"❌ GPS update failed: {output}")
                    else:
                        st.error(
                            "Please provide image folder, CSV path, and output directory"
                        )

    # Tab 4: Visualization
    with tab4:
        st.header("👁️ Visualization")

        # Visualize Detection
        st.subheader("Visualize Detection Dataset")
        with st.expander("Visualize Detection", expanded=True):
            detection_dataset_name = st.text_input(
                "Dataset Name", key="detection_dataset"
            )
            detection_root = st.text_input("Root Data Directory", key="detection_root")
            detection_split = st.selectbox(
                "Split", ["train", "val", "test"], key="detection_split"
            )

            if st.button("Visualize Detection"):
                if detection_dataset_name and detection_root:
                    args = [
                        detection_dataset_name,
                        "--root",
                        detection_root,
                        "--split",
                        detection_split,
                    ]
                    success, output = run_cli_command("visualize-detection", args)
                    if success:
                        st.success("✅ Detection visualization launched!")
                    else:
                        st.error(f"❌ Detection visualization failed: {output}")
                else:
                    st.error("Please provide dataset name and root directory")

        # Visualize Classification
        st.subheader("Visualize Classification Dataset")
        with st.expander("Visualize Classification", expanded=True):
            classification_dataset_name = st.text_input(
                "Dataset Name", key="classification_dataset"
            )
            classification_root = st.text_input(
                "Root Data Directory", key="classification_root"
            )
            classification_split = st.selectbox(
                "Split", ["train", "val", "test"], key="classification_split"
            )

            # Optional parameters
            load_as_single_class = st.checkbox(
                "Load as Single Class", key="single_class"
            )
            background_class = st.text_input(
                "Background Class", value="background", key="background_class"
            )
            single_class_name = st.text_input(
                "Single Class Name", value="wildlife", key="single_class_name"
            )
            keep_classes = st.text_input(
                "Keep Classes (comma-separated)", key="keep_classes"
            )
            discard_classes = st.text_input(
                "Discard Classes (comma-separated)", key="discard_classes"
            )

            if st.button("Visualize Classification"):
                if classification_dataset_name and classification_root:
                    args = [
                        classification_dataset_name,
                        "--root",
                        classification_root,
                        "--split",
                        classification_split,
                    ]

                    if load_as_single_class:
                        args.extend(["--single-class"])
                    if background_class:
                        args.extend(["--background-class", background_class])
                    if single_class_name:
                        args.extend(["--single-class-name", single_class_name])
                    if keep_classes:
                        args.extend(["--keep-classes", keep_classes])
                    if discard_classes:
                        args.extend(["--discard-classes", discard_classes])

                    success, output = run_cli_command("visualize-classification", args)
                    if success:
                        st.success("✅ Classification visualization launched!")
                    else:
                        st.error(f"❌ Classification visualization failed: {output}")
                else:
                    st.error("Please provide dataset name and root directory")

    # Tab 5: Dataset Management
    with tab5:
        st.header("📊 Dataset Management")

        st.subheader("List Datasets")
        with st.expander("List Datasets", expanded=True):
            list_root = st.text_input("Root Directory", value="data", key="list_root")
            list_verbose = st.checkbox("Verbose Output", key="list_verbose")

            if st.button("List Datasets"):
                args = ["--root", list_root]
                if list_verbose:
                    args.append("--verbose")

                success, output = run_cli_command("list-datasets", args)
                if success:
                    st.success("✅ Datasets listed successfully!")
                    st.code(output)
                else:
                    st.error(f"❌ Failed to list datasets: {output}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>WildData Pipeline UI - Powered by Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
