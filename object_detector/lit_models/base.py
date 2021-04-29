import argparse
import pytorch_lightning as pl
import torch

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100

class Accuracy(pl.metrics.Accuracy):

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:

        """
        Pytorch-lightning 1.2 이상 버전의 메트릭은 preds가 0에서 1 사이일 것으로 예상하며
        그렇지 않으면 ValueError가 발생하고 실패합니다.

        torch.nn 과 torch.nn.fucntional은 대체로 비슷하지만 사용법이 조금씩 다름.
        """

        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=1)
        super.update(preds=preds, target=target)

class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()

        # self.model에 전달받은 model attribute 설정
        self.model = model
        # vars() : 모듈, 클래스, 인스턴스 또는 __dict__ 어트리뷰트가 있는 다른 객체의 __dict__ 어트리뷰트를 돌려줍니다.
        # args로 무엇인가 받아서 None이 아니면 vars(args) 수행, 아니면 빈 dict 설정.
        self.args = vars(args) if args is not None else {}

        # 명령행에서 전달받은 dict에 optimizer라는 key값이 있으면 그 value로 설정 / 없으면 기본 OPTIMIZER 값으로 설정.
        optimizer = self.args.get("optimizer", OPTIMIZER)
        # getattr : object의 속성값을 가져올 때 사용(torch.optim의 optimizer 속성값)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        if loss not in ("cross_entropy", "softmax"):
            self.loss_fn = getattr(torch.nn.functional, loss)
        
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizezr", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=str, defalut=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer" : optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def foward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        training_step에서는 One batch에 대한 손실을 반환하는 과정을 정의함.
        이 과정은 training 및 inference 과정에서 자동으로 반복됨.
        """

        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss