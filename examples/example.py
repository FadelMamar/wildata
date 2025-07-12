# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:57:28 2025

@author: FADELCO
"""

from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.pipeline.path_manager import PathManager
from wildtrain.transformations import TransformationPipeline, TilingTransformer, AugmentationTransformer, BoundingBoxClippingTransformer

ROOT = r"D:\workspace\data\demo-dataset"
SOURCE_PATH = r"D:\workspace\savmap\coco\annotations\train.json"
def main():

    trs = TransformationPipeline()
    trs.add_transformer(BoundingBoxClippingTransformer(tolerance=5,skip_invalid=True))
    trs.add_transformer(AugmentationTransformer())
    #trs.add_transformer(TilingTransformer())

    pipeline = DataPipeline(root=ROOT,transformation_pipeline=trs)
    pipeline.import_dataset(
        source_path=SOURCE_PATH,
        source_format="coco",
        dataset_name="demo",
        apply_transformations=True
    )


if __name__ == "__main__":
    main()


