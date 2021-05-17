import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from object_detector.utils.utils import provide_determinism, load_classes
from object_detector.models import Darknet, load_model
from object_detector.utils.parse_cfg import parse_data_config
from object_detector.utils.loss import compute_loss

from torchsummary import summary

class Accuracy(pl.metrics.Accuracy):

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:

        """
        Pytorch-lightning 1.2 이상 버전의 메트릭은 preds가 0에서 1 사이일 것으로 예상하며
        그렇지 않으면 ValueError가 발생하고 실패합니다.

        torch.nn 과 torch.nn.fucntional은 대체로 비슷하지만 사용법이 조금씩 다름.
        """

        if preds.min() < 0 or preds.max() > 1:
            preds = F.softmax(preds, dim=1)
        super.update(preds=preds, target=target)


class LitYoloModule(pl.LightningModule):

    def __init__(self, args: argparse.Namespace = None, model = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        if self.args.get("seed") != -1:
            provide_determinism(self.args.get("seed"))

        ##############
        # Create model
        ##############
        self.model = model

        # Print model
        if self.args.get("verbose"):
            summary(self.model, input_size=(3, self.model.hyperparams['height'], self.model.hyperparams['height']))

        ##################
        # Create optimizer
        ##################

        self.params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer_class = self.model.hyperparams['optimizer']
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
        parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
        parser.add_argument("--verbose", action='store_true', help="Makes the training more verbose")
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer_class in [None, "adam"]:
            optimizer = optim.Adam(
                self.params,
                lr=self.model.hyperparams['learning_rate'],
                weight_decay=self.model.hyperparams['decay']
            )
        elif self.optimizer_class == "sgd":
            optimizer = optim.SGD(
                self.params,
                lr=self.model.hyperparams['learning_rate'],
                weight_decay=self.model.hyperparams['decay'],
                momentum=self.model.hyperparams['momentum']
            )
        else:
            print("Unknown optimzier. Please choose between (adam, sgd).")
        
        return optimizer

    def training_step(self, train_batch, batch_idx):
        _, x, y = train_batch
        z = self.model(x)
        loss, loss_components = compute_loss(z, y, self.model)
        self.log("train_loss", loss, on_epoch=True)
        self.train_acc(z, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        _, x, y = val_batch
        z = self.model(x)
        loss, loss_components = compute_loss(z, y, self.model)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(z, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, test_batch, batch_idx):
        _, x, y = test_batch
        z = self.model(x)
        self.test_acc(z, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)