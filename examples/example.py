# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:57:28 2025

@author: FADELCO
"""

from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.pipeline.path_manager import PathManager

ROOT = r"D:\workspace\data\MyNewData"

def main():
    pipeline = DataPipeline(root=ROOT)
    pipeline.import_dataset(
        source_path=r"D:\workspace\savmap\coco\annotations\train.json",
        source_format="coco",
        dataset_name="demo",
    )


if __name__ == "__main__":
    main()


