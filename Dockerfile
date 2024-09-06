FROM quay.io/azavea/raster-vision:pytorch-latest

RUN pip install pytorch-lightning==2.* timm einops terratorch

CMD ["bash"]