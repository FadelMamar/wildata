@echo off
REM Visualize datasets using the wildata CLI
REM This script provides examples for both classification and detection visualization

call cd /d "%~dp0" && cd ..
echo ========================================
echo WildData Visualization Examples
echo ========================================

REM Set the dataset name and root directory (edit as needed)
set DATASET_NAME=savmap
set ROOT_DIR=D:\workspace\data\demo-dataset

REM Optionally set other parameters (uncomment and edit as needed)
REM set KEEP_CLASSES=lion,elephant
REM set DISCARD_CLASSES=termite mound,rocks
REM set SPLIT=val

echo.
echo Example 1: Visualize Classification Dataset
echo -------------------------------------------
echo Dataset: %DATASET_NAME%
echo Root Directory: %ROOT_DIR%
echo Split: train
echo.

echo call uv run wildata visualize-classification %DATASET_NAME% --root %ROOT_DIR% --split train

echo.
echo Example 2: Visualize Classification Dataset with Class Filtering
echo ----------------------------------------------------------------
echo Dataset: %DATASET_NAME%
echo Root Directory: %ROOT_DIR%
echo Split: val
echo Keep Classes: 
echo Discard Classes: termite mound,rocks
echo.

REM Example with keep/discard classes (uncomment to use)
echo call uv run wildata visualize-classification %DATASET_NAME% --root %ROOT_DIR% --split val --keep-classes %KEEP_CLASSES% --discard-classes %DISCARD_CLASSES%

echo.
echo Example 3: Visualize Detection Dataset
echo --------------------------------------
echo Dataset: %DATASET_NAME%
echo Root Directory: %ROOT_DIR%
echo Split: train
echo.

REM Run the visualize_detection command
call uv run wildata visualize-detection %DATASET_NAME% --root %ROOT_DIR% --split train

echo.
echo ========================================
echo Visualization Examples Complete
echo ========================================
echo.
echo To use different parameters:
echo 1. Edit the variables at the top of this script
echo 2. Uncomment the desired example lines
echo 3. Comment out the example you don't want to run
echo.
echo Available options:
echo - --split: train, val, or test
echo - --keep-classes: comma-separated list of classes to keep
echo - --discard-classes: comma-separated list of classes to discard
echo - --single-class: true/false for classification
echo - --background-class: background class name
echo - --single-class-name: single class name
echo.

call pause