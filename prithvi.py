import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from terratorch.models import PrithviModelFactory
from terratorch.datasets import HLSBands
from torchmetrics import JaccardIndex
from sklearn.metrics import precision_recall_fscore_support

class PrithviSemanticSegmentation(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        in_channels,
        num_frames,
        decoder_num_convs,
        img_size,
        learning_rate,
    ):
        super().__init__()
        model_factory = PrithviModelFactory()
        self.model = model_factory.build_model(
            task="segmentation",
            backbone="prithvi_vit_100",
            decoder="FCNDecoder",
            decoder_num_convs=decoder_num_convs,
            in_channels=in_channels,
            bands=[
                HLSBands.BLUE,
                HLSBands.GREEN,
                HLSBands.RED,
                HLSBands.NIR_NARROW,
                HLSBands.SWIR_1,
                HLSBands.SWIR_2,
            ],
            num_classes=num_classes,
            pretrained=True,
            num_frames=num_frames,
            head_dropout=0.0,
            img_size=img_size,
        )
        self.learning_rate = learning_rate

        for param in self.model.encoder.parameters():
            param.requires_grad = False

        self.jaccard_index = JaccardIndex(task="multiclass", num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        model_output = self.model(x)
        mask = model_output.output
        loss = F.cross_entropy(mask, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        pred = torch.argmax(mask, dim=1)
        iou = self.jaccard_index(pred, y)
        self.log("train/iou", iou, on_step=False, on_epoch=True)
        y_flat = y.flatten().cpu()
        pred_flat = pred.flatten().cpu()
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_flat, pred_flat, average="macro"
        )
        self.log_dict(
            {
                "train/precision_macro": precision_macro,
                "train/recall_macro": recall_macro,
                "train/f1_macro": f1_macro,
            },
            on_step=False,
            on_epoch=True,
        )

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(y_flat, pred_flat, average="weighted")
        )
        self.log_dict(
            {
                "train/precision_weighted": precision_weighted,
                "train/recall_weighted": recall_weighted,
                "train/f1_weighted": f1_weighted,
            },
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        model_output = self.model(x)
        mask = model_output.output
        loss = F.cross_entropy(mask, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        pred = torch.argmax(mask, dim=1)
        iou = self.jaccard_index(pred, y)
        self.log("val/iou", iou, on_step=False, on_epoch=True)
        y_flat = y.flatten().cpu()
        pred_flat = pred.flatten().cpu()
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_flat, pred_flat, average="macro"
        )
        self.log_dict(
            {
                "val/precision_macro": precision_macro,
                "val/recall_macro": recall_macro,
                "val/f1_macro": f1_macro,
            },
            on_step=False,
            on_epoch=True,
        )

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(y_flat, pred_flat, average="weighted")
        )
        self.log_dict(
            {
                "val/precision_weighted": precision_weighted,
                "val/recall_weighted": recall_weighted,
                "val/f1_weighted": f1_weighted,
            },
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer