import argparse

import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from object_detector import lit_models, data

np.random.seed(42)
torch.manual_seed(42)

CHECKPOINT_DIR = '../Object_detection_utils/checkpoints/'
LOG_DIR = '../Object_detection_utils/'

def _set_up_parser():

    parser = argparse.ArgumentParser(add_help=False)

    # Trainer arguments
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.LitYoloModule.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser

def main():

    parser = _set_up_parser()

    args = parser.parse_args()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=LOG_DIR, name="yolo_lightning_logs"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=CHECKPOINT_DIR,
        filename='yolo-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min'
    )

    datamodule = data.LitDataModule

    lit_model_class = lit_models.LitYoloModule

    lit_model = lit_model_class(args=args)

    trainer = Trainer.from_argparse_args(args, default_root_dir=CHECKPOINT_DIR, callbacks=[checkpoint_callback], logger=[tb_logger])

    trainer.fit(lit_model, datamodule=datamodule)

    datamodule.setup(stage="test")
    trainer.test(lit_model, datamodule=datamodule)

if __name__ == "__main__":
    main()