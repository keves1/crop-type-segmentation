## Crop Type Segmentation with Multi-temporal Landsat Imagery

This project demonstrates semantic segmentation of satellite imagery for crop type identification. Because of the seasonal patterns present in crop growth, the use of time series can improve crop type classification results. In this project, I use the frozen pretrained encoder from the [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) foundation model and train a segmentation head for this task using Landsat 9 imagery (6 bands) as inputs and the Cropland Data Layer (CDL) as labels. The time series are monthly composites across the growing season for a single year.

### Container Environment Setup

After cloning this repo, build a Docker image using the included Dockerfile:
```bash
cd crop-type-segmentation
docker build -t crop-type-segmentation:latest .
```

Use the Dev Containers extension in VSCode to open the code in the container (see `.devcontainer` folder).

### Data

Both the Landsat 9 and CDL data can be obtained from Google Earth Engine by running the script [here](https://code.earthengine.google.com/808172f2afcfc353e5f34003aafe6ea4?noload=true).

The data should be placed in `crop-type-segmentation/data`.

### Training

To train the model, run `train.ipynb`. 