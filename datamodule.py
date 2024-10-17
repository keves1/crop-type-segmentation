from pathlib import Path
import re
import math

import pytorch_lightning as pl
from rastervision.core.data import (
    RasterioSource,
    MinMaxTransformer,
    TemporalMultiRasterSource,
    Scene,
    SemanticSegmentationLabelSource,
    ClassConfig,
    NanTransformer,
    ReclassTransformer,
)
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset
from shapely.geometry import Polygon
import torch
from torch.utils.data import DataLoader
import albumentations as A

from cropland_data_layer_class_table import class_info


class CropTypeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        months: list,
        channels: list,
        train_percent: float,
        img_size: int,
        batch_size: int,
        num_workers: int,
        use_randomrotate90: bool = False,
        use_flip: bool = False,
        use_transpose: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.months = months
        self.channels = channels
        self.train_percent = train_percent
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_randomrotate90 = use_randomrotate90
        self.use_flip = use_flip
        self.use_transpose = use_transpose

    def setup(self, stage) -> None:
        months_regex = (
            f"Landsat9_Composite_2022_0[{''.join(map(str, self.months))}].tiff"
        )
        l9_images = sorted(self.data_dir.glob("Landsat9_Composite_2022_0*.tiff"))
        l9_images = [img for img in l9_images if re.match(months_regex, img.name)]

        colors = [item["Color"] for item in class_info]
        names = [item["Description"] for item in class_info]

        # Map class IDs to use classes that contain more than 1% of pixels. All other classes are "Other" (0).
        # All classes for developed areas are combined
        most_frequent_crops = {
            3: 1,
            6: 2,
            24: 3,
            36: 4,
            37: 5,
            54: 6,
            61: 7,
            75: 8,
            76: 9,
            111: 10,
            142: 11,
            152: 12,
            176: 13,
            195: 14,
            220: 15,
        }
        developed_classes = [82, 121, 122, 123, 124]
        mapping = {}
        for item in class_info:
            value = int(item["Value"])
            if value in most_frequent_crops:
                mapping[value] = most_frequent_crops[value]
            elif value in developed_classes:
                mapping[value] = 16
            else:
                mapping[value] = 0

        class_config = ClassConfig(names=names, colors=colors, null_class="Other")
        label_source = SemanticSegmentationLabelSource(
            raster_source=RasterioSource(
                uris="data/Cropland_Data_Layer_2022.tiff",
                raster_transformers=[ReclassTransformer(mapping)],
            ),
            class_config=class_config,
        )
        raster_sources = []

        for image_uri in l9_images:
            raster_sources.append(
                RasterioSource(
                    str(image_uri),
                    channel_order=self.channels,
                    raster_transformers=[
                        NanTransformer(to_value=0),
                        MinMaxTransformer(),
                    ],
                )
            )

        time_series = TemporalMultiRasterSource(raster_sources)

        extent = raster_sources[0].bbox.extent
        extent = extent.to_dict()

        train_aoi = Polygon.from_bounds(
            ymin=0,
            ymax=int(extent["ymax"] * self.train_percent),
            xmin=0,
            xmax=extent["xmax"],
        )
        val_aoi = Polygon.from_bounds(
            ymin=math.ceil(extent["ymax"] * self.train_percent),
            ymax=extent["ymax"],
            xmin=0,
            xmax=extent["xmax"],
        )

        train_scene = Scene(
            id="train",
            raster_source=time_series,
            label_source=label_source,
            aoi_polygons=[train_aoi],
        )
        val_scene = Scene(
            id="val",
            raster_source=time_series,
            label_source=label_source,
            aoi_polygons=[val_aoi],
        )

        train_transforms = []
        if self.use_randomrotate90:
            train_transforms.append(A.RandomRotate90(p=0.5))
        if self.use_flip:
            train_transforms.append(A.HorizontalFlip(p=0.5))
            train_transforms.append(A.VerticalFlip(p=0.5))
        if self.use_transpose:
            train_transforms.append(A.Transpose(p=0.5))

        train_transforms = A.Compose(train_transforms)

        self.train_dataset = SemanticSegmentationSlidingWindowGeoDataset(
            train_scene,
            size=self.img_size,
            stride=self.img_size,
            padding=0,
            transform=train_transforms,
        )
        self.val_dataset = SemanticSegmentationSlidingWindowGeoDataset(
            val_scene, size=self.img_size, stride=self.img_size, padding=0
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
        )

    @staticmethod
    def custom_collate_fn(batch):
        """Changes the order of the axes from what Raster Vision outputs (B,T,C,H,W) to what
        the Prithvi model expects (B,C,T,H,W).
        """
        data, targets = zip(*batch)
        data = torch.stack(data)
        data = data.permute(0, 2, 1, 3, 4)
        if isinstance(targets[0], torch.Tensor):
            targets = torch.stack(targets)
        else:
            targets = torch.tensor(targets)
        return data, targets
