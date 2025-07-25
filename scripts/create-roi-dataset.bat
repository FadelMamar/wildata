@echo off
REM Example: Import a dataset using the WildTrain CLI with a YAML config file

REM Set the path to your config file (edit as needed)
set CONFIG_FILE=configs\roi-create-config.yaml

REM Run the import command using only the config file
call wildata create-roi-dataset --config %CONFIG_FILE%