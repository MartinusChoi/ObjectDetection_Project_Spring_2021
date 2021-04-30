['Lightning in 2 steps'](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)의 내용을 기반으로 하고 있습니다.

## PyTorch Lightning으로 code를 Organizing 하면 :

- research code를 engineering에서 분리하여 더욱 읽기 쉽게 만들어 줍니다.
- reproduce하기에 더욱 쉬워집니다.
- 대부분의 training loop 및 까다로운 engineering을 자동화하여 오류 발생률을 감소시켜 줍니다.
- model의 변경 없이 모든 하드웨어로 확장이 가능합니다.

# Organize *PyTorch code* to *PyTorch Lightning*

- 아래의 PyTorch code를 PyTorch Lightning으로 Organizing 해봅시다.
  ~~~python
  # models
  self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
  self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))
      
  encoder.cuda(0)
  decoder.cuda(0)
  
  # download on rank 0 only
  if global_rank == 0:
    mnist_train = MNIST(os.getcwd(), train=True, download=True)
  
  # split dataset
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(0.5, 0.5)])
  mnist_train = MNIST(os.getcwd(), train=True, download=True, transoform=transform)
  
  # train (55,000 images), val split (5,000 images)
  mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
  
  # The dataloaders handle shuffling, batching, etc ...
  mnist_train = DataLoader(mnist_train, batch_size=64)
  mnist_cal = DataLoader(mnist_val, batch_size=64)
  
  # optimizer
  params = [encoder.parameters(), decoder.parameters()]
  optimizer = torch.optim.Adam(params, lr=1e-3)
  
  # TRAIN LOOP
  model.train()
  num_epoch = 1
  for epoch in range(num_epochs):
    for train_batch in mnist_train:
      x, y = train_batch
      x = x.cuda(0)
      x = x.view(x.size(0), -1)
      z = encoder(x)
      x_hat = decoder(z)
      loss = F.mse_loss(x_hat, x)
      print('train loss: ', loss.item())
    
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    
  # EVAL LOOP
  model.eval()
  with torch.no_grad():
    val_loss = []
    for val_batch in mnist_val:
      x, y = val_batch
      x = x.cuda(0)
      x = x.view(x.size(0), -1)
      z = encoder(x)
      x_hat = decoder(z)
      loss = F.mse_loss(x_hat, x)
      val_loss.append(loss)
    val_loss = torch.mean(torch.tensor(val_loss))
    model.train()
    ~~~

## Step 0 : Install PyTorch Lightning

### using pip

~~~
pip install pytorch-lightning
~~~

### using conda

~~~
conda install pytorch-lightning -c conda-forge
~~~

### can use conda environment

~~~
conda activate my_env
pip install pytorch-lightning
~~~

### import the following :

~~~python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transform
from torchvision.datasets import MNIST
from torchvision.data import DataLoader, random_split
import pytorch_lightning as pl
~~~

## Step 1 : LightningModule 정의

### 1. Computational code를 LightningModule에 넣는다.
- Model architecture를 __init__ 안으로.
  - PyTorch
    ~~~python
    # models
    self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))
      
    encoder.cuda(0)
    decoder.cuda(0)
    ~~~
  - PyTorch Lightning
    ~~~python
    class LitAutoEncoder(pl.LightningModule):
      def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))
        
        encoder.cuda(0)
        encoder.cuda(0)
    ~~~

### 2. foward hook 설정.

- lightning에서는 **foward**는 **prediction/inference action**을 정의한다.
  - PyTorch Lightning
    ~~~python
    class LitAutoEncoder(pl.LightningModule):
      def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))
        
        encoder.cuda(0)
        encoder.cuda(0)
        
      def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    ~~~

### 3. Optimizers를 configure_optimizer 안으로 넣는다.
- configure_optimizer => optimizer를 정의하는 LightningModule hook
- Self.parameters 가 __init__ 에서 정의한 model의 parameter 들을 가지고 있다.
  - PyTorch
    ~~~python
    # optimizer
    params = [encoder.parameters(), decoder.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    ~~~
  - PyTorch Lightning
    ~~~python
    class LitAutoEncoder(pl.LightningModule):
      def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))
        
        encoder.cuda(0)
        encoder.cuda(0)
        
      def forward(self, x):
        embedding = self.encoder(x)
        return embedding
        
      def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer
    ~~~

### 4. Training logic을 trainig_step 안으로 넣는다.
- training_step => training 과정에서 iteration 1번에서의 동작을 정의하는 LightningModule hook
- 나중에 training에 이용하고 싶으면, loss나 predictions의 dictionary를 반환시킬 수 있다.
  - PyTorch
    ~~~python
    x, y = train_batch
    x = x.cuda(0)
    x = x.view(x.size(0), -1)
    z = encoder(x)
    x_hat = decoder(z)
    loss = F.mse_loss(x_hat, x)
    ~~~
  - PyTorch Lightning
    ~~~python
    class LitAutoEncoder(pl.LightningModule):
      def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))
        
        encoder.cuda(0)
        encoder.cuda(0)
        
      def forward(self, x):
        embedding = self.encoder(x)
        return embedding
        
      def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer
        
      def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        """
        TensorBoard나 다른 logger에 metric을 보내고 싶다면, self.log를 사용할 수 있다.
        epoch-level에서 metric을 계산하고 싶으면, self.log에 on_epoch=True를 추가한다.
        """
        self.log('train_loss', loss, on_epoch=True)
        return loss
    ~~~
       
### 5. Validation logic을 validation_step 안으로 넣는다.
- validation_step => validation 과정에서의 동작을 정의하는 LightningModule hook
  - PyTorch
    ~~~python
    x, y = val_batch
    x = x.cuda(0)
    x = x.view(x.size(0), -1)
    z = encoder(x)
    x_hat = decoder(z)
    loss = F.mse_loss(x_hat, x)
    ~~~
  - PyTorch Lightning
    ~~~python
    class LitAutoEncoder(pl.LightningModule):
      def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))
        
        encoder.cuda(0)
        encoder.cuda(0)
        
      def forward(self, x):
        embedding = self.encoder(x)
        return embedding
        
      def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer
        
      def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        """
        TensorBoard나 다른 logger에 metric을 보내고 싶다면, self.log를 사용할 수 있다.
        epoch-level에서 metric을 계산하고 싶으면, self.log에 on_epoch=True를 추가한다.
        """
        self.log('train_loss', loss, on_epoch=True)
        return loss
        
      def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        """
        validation_step hook에서 self.log를 호출하면,
        자동으로 epoch의 마지막에 log를 축적한다.
        """
        self.log('val_loss', loss)
    ~~~
    
### 6. .cuda() 와 같은 모든 device calls를 지운다.
- LightningModule은 hardware에 대해 불가지론적이다. (즉, device와 관련한 처리를 하지 않는다.)
  - PyTorch Lightning
    ~~~python
    x, y = val_batch
    x = x.cuda(0)
    x = x.view(x.size(0), -1)
    z = encoder(x)
    x_hat = decoder(z)
    loss = F.mse_loss(x_hat, x)
    ~~~
  - PyTorch Lightning
    ~~~python
    class LitAutoEncoder(pl.LightningModule):
      def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))
        
      def forward(self, x):
        embedding = self.encoder(x)
        return embedding
        
      def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer
        
      def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        """
        TensorBoard나 다른 logger에 metric을 보내고 싶다면, self.log를 사용할 수 있다.
        epoch-level에서 metric을 계산하고 싶으면, self.log에 on_epoch=True를 추가한다.
        """
        self.log('train_loss', loss, on_epoch=True)
        return loss
        
      def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        """
        validation_step hook에서 self.log를 호출하면,
        자동으로 epoch의 마지막에 log를 축적한다.
        """
        self.log('val_loss', loss)
    ~~~
    
### (Optional) 7. 필요한 LightningModule hook를 Override 한다.
- LightningModule은 20개 이상의 hooks들을 유지할 수 있다.
  - PyTorch
    ~~~python
    loss.backward()
    optimizer.step()
    ~~~
  - PyTorch Lightning
    ~~~python
    class LitAutoEncoder(pl.LightningModule):
      def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))
        
      def forward(self, x):
        embedding = self.encoder(x)
        return embedding
        
      def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer
        
      def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        """
        TensorBoard나 다른 logger에 metric을 보내고 싶다면, self.log를 사용할 수 있다.
        epoch-level에서 metric을 계산하고 싶으면, self.log에 on_epoch=True를 추가한다.
        """
        self.log('train_loss', loss, on_epoch=True)
        return loss
        
      def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        """
        validation_step hook에서 self.log를 호출하면,
        자동으로 epoch의 마지막에 log를 축적한다.
        """
        self.log('val_loss', loss)
        
      def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
        
      def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
    ~~~

## Step 2 : Fit with Lightning Trainer

### 0. DataLoader를 정의한다.
~~~python
# download on rank 0 only
if global_rank == 0:
  mnist_train = MNIST(os.getcwd(), train=True, download=True)

# split dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])
mnist_train = MNIST(os.getcwd(), train=True, download=True, transoform=transform)

# train (55,000 images), val split (5,000 images)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

# The dataloaders handle shuffling, batching, etc ...
mnist_train = DataLoader(mnist_train, batch_size=64)
mnist_cal = DataLoader(mnist_val, batch_size=64)
~~~

### 1. 정의한 LightningModule을 init한다.
~~~python
model = LitAutoEncoder()
~~~
- Lightning trainer는 다음의 engineering을 자동화 한다.
  - loops
  - hardware calls
  - model.train()
  - model.eval()
  - zero_gard

### 2. Lightning Trainer를 init한다.
~~~python
model = LitAutoEncoder()
trainier = pl.Trainer()
~~~

### 3. PyTorch DataLoader를 전달한다.
~~~python
model = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(model, mnist_train, mnist_val)
~~~

### (Optional) Add any other functionality using our Callbacks API
- Callbacks는 임의의 code를 정확한 시간에 실행시키기 위한 self contained progess 이다.
~~~python
callbacks = [SSLOnlineEvaluator(), ConfusedLogitCallback()]
model = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(model, mnist_train, mnist_val)
~~~

### Train as fast as lightning
- model을 바꾸지 않고도, 여러개의 GPUs와 TPUs를 사용하여 train할 수 있다.

#### Trian on CPUs
~~~python
model = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(model, mnist_train, mnist_val)
~~~

#### Train on GPUs
~~~python
model = LitAutoEncoder()
trainer = pl.Trainer(gpus=4)
trainer.fit(model, mnist_train, mnist_val)
~~~

### Train on TPUs
~~~python
model = LitAutoEncoder()
trainer = pl.Trainer(tpu_cores=8)
trainer.fit(model, mnist_train, mnist_val)
~~~
