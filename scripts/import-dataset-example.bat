@echo off
REM Example: Import a dataset using the WildTrain CLI with a YAML config file

REM Set the path to your config file (edit as needed)
set CONFIG_FILE=example-import-config.yaml

REM Run the import command using only the config file
uv run python -m wildtrain dataset import --config %CONFIG_FILE%

REM Run the import command with CLI overrides (e.g., override dataset_name)
uv run python -m wildtrain dataset import --config %CONFIG_FILE% --dataset-name my_overridden_dataset

REM Run the import command using only CLI arguments (no config file)
uv run python -m wildtrain dataset import --source-path "data/mydata.json" --format-type coco --dataset-name cli_only_dataset --augment true --tile true 