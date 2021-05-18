import argparse
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from object_detector.utils.utils import provide_determinism, load_classes, xywh2xyxy, to_cpu, non_max_suppression, get_batch_statistics, ap_per_class, print_eval_stats
from object_detector.models import Darknet, load_model
from object_detector.utils.parse_cfg import parse_data_config
from object_detector.utils.loss import compute_loss

from torchsummary import summary


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
        self.verbose = self.args.get("verbose")
        if self.verbose:
            summary(self.model, input_size=(3, self.model.hyperparams['height'], self.model.hyperparams['height']))

        # validation requires
        self.conf_thres = self.args.get("conf_thres")
        self.nms_thres = self.args.get("nms_thres")
        self.iou_thres = self.args.get("iou_thres")
        data_config = parse_data_config(self.args.get("data"))
        self.class_names = data_config["names"]

        ##################
        # Create optimizer
        ##################

        self.params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer_class = self.model.hyperparams['optimizer']

        self.labels = list()
        self.sample_metrics = list()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
        parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
        parser.add_argument("--verbose", action='store_true', help="Makes the training more verbose")
        parser.add_argument("--conf_thres", type=float, default=0.01, help="Obejct confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
        parser.add_argument("--iou_thres", type=float, default=0.5, help="IOY threshold required to qulify as detected")
    
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
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        _, x, y = val_batch
        
        # Extract labels
        self.labels += y[:, 1].tolist()
        
        # Rescale target
        y[:, 2:] = xywh2xyxy(y[:, 2:])
        y[:, 2:] *= x.shape[2]

        x = Variable(x, requires_grad=False)

        with torch.no_grad():
            z = to_cpu(self.model(x))
            z = non_max_suppression(z, conf_thres=self.conf_thres, iou_thres=self.nms_thres)

        self.sample_metrics += get_batch_statistics(z, y, iou_threshold=self.iou_thres)
    
    
    def validation_step_end(self, *args, **kwargs):
        if len(self.sample_metrics) == 0:
            print("---- No detections over batch validation set ----")
            return None
        
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*self.sample_metrics))
        ]
        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_scores, self.labels
        )

        print_eval_stats(metrics_output, self.class_names, self.verbose)

        _, _, AP, _, _ = metrics_output

        self.log("val_AP", AP, prog_bar=True)

        return AP
        
    def test_step(self, test_batch, batch_idx):
        _, x, y = test_batch
        
        # Extract labels
        self.labels += y[:, 1].tolist()
        
        # Rescale target
        y[:, 2:] = xywh2xyxy(y[:, 2:])
        y[:, 2:] *= x.shape[2]

        x = Variable(x, requires_grad=False)

        with torch.no_grad():
            z = to_cpu(self.model(x))
            z = non_max_suppression(z, conf_thres=self.conf_thres, iou_thres=self.nms_thres)

        self.sample_metrics += get_batch_statistics(z, y, iou_threshold=self.iou_thres)
    
    def test_step_end(self, *args, **kwargs):
        if len(self.sample_metrics) == 0:
            print("---- No detections over batch validation set ----")
            return None
        
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*self.sample_metrics))
        ]
        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_scores, self.labels
        )

        print_eval_stats(metrics_output, self.class_names, self.verbose)

        _, _, AP, _, _ = metrics_output

        self.log("test_AP", AP, prog_bar=True)

        return AP