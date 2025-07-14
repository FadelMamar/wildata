# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:57:28 2025

@author: FADELCO
"""

from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.pipeline.path_manager import PathManager
from wildtrain.transformations import TransformationPipeline, TilingTransformer, AugmentationTransformer, BoundingBoxClippingTransformer
from wildtrain.config import ROOT,ROIConfig

ROOT_DATA = r"D:\workspace\data\demo-dataset"
# SOURCE_PATH = r"D:\workspace\savmap\coco\annotations\train.json"
SOURCE_PATH = r"D:\workspace\data\project-4-at-2025-07-14-10-55-95d5eea7.json" 

def main():

    trs = TransformationPipeline()
    trs.add_transformer(BoundingBoxClippingTransformer(tolerance=5,skip_invalid=True))
    trs.add_transformer(AugmentationTransformer())
    trs.add_transformer(TilingTransformer())

    pipeline = DataPipeline(root=ROOT_DATA,
                            transformation_pipeline=trs,
                            split_name="train")
    pipeline.import_dataset(
        source_path=SOURCE_PATH,
        source_format="ls",
        dataset_name="demo",
        ls_parse_config=False,
        ls_xml_config=str(ROOT / "configs" / "label_studio_config.xml"),
        dotenv_path="../.env",
        roi_config=ROIConfig(
            random_roi_count=10,
            roi_box_size=128,
            min_roi_size=32,
            background_class="background",
            save_format="jpg",
        )
    )


if __name__ == "__main__":
    main()


