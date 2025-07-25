@echo off
REM Visualize classification data using the wildata CLI

REM Set the dataset name and root directory (edit as needed)
set DATASET_NAME=dryseason-kapiri-camp9-11-rep1
set ROOT_DIR=D:/PhD/workspace/data

REM Optionally set other parameters (uncomment and edit as needed)
REM set KEEP_CLASSES=lion,elephant
REM set DISCARD_CLASSES=termite mound,rocks
REM set SPLIT=val

REM Run the visualize_classification command
call uv run wildata visualize-classification %DATASET_NAME% --root %ROOT_DIR% --split train
REM To add keep/discard classes, add:
REM call uv run wildata visualize-classification %DATASET_NAME% --root %ROOT_DIR% --split val --keep-classes %KEEP_CLASSES% --discard-classes %DISCARD_CLASSES%
