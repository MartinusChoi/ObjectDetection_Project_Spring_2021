import argparse
 
import torch
from pytorch_lightning import LightningModule

from Detect.models.yolo import Model

class LitYoloModle(LightningModule):
    def __init__(self, args:argparse.Namespace = None, model=None):
        super().__init__()

        self.args = vars(args) if args is not None else {}
        self.model = model



    
    @staticmethod
    def add_to_argparse(parse):
        parse.add_argument("weights", type=str, default="yolov3.weights")
        parse.add_argument("cfg", type=str, default="yolov3.cfg")
        return parse
    
    def forward(self, x):
        return