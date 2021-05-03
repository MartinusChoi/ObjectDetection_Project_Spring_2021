- 아래의 lightning의 features들은 **Trainer** 혹은 **LightningModule**에 관한 것입니다.

***************************************************************************************

# Basic Features

## Manual vs. automatic optimization

### Automatic optimization

- **Lightning**을 활용하면 PyTorch에서 처럼 **grad를 enable/disable**하거나, **backward pass**를 하거나, *training_step*에서 attach된 graph와 함께 loss를 반환받는 한 **optimizers를 update** 하는 것에 대해서 걱정할 필요가 없습니다.
- Lightning은**opimization을 자동화** 하기 때문입니다.
  ~~~python
  def training_step(self, batch, batch_idx):
    loss = self.encoder(batch)
    return loss
  ~~~
  
### Manual optimization

- 하지만 GAN, 강화 학습 또는 여러 optimizer 또는 inner loop를 사용하는 것과 같은 특정 연구의 경우 automatic optimization을 끄고 Training loop를 직접 완전히 제어할 수 있습니다.
  ~~~python
  def __init(self):
    # automatic optimization을 끄는 명령어
    self.automatic_optimization = False
  
  def training_step(self, batch, batch_idx):
    # use_pl_optimizer=False 의 명령어로 optimizer에 접근합니다. (default=True)
    opt_a, opt_b = self.optimizers(use_pl_optimizer=True)
    
    loss_a = self.generator(batch)
    opt_a.zero_grad()
    # 'loss.backward'대신 'manual_backward()'를 사용하여 half precision 등.. 과 같은 작업을 자동화 합니다.
    self.manual_backward(loss_a)
    opt_a.step()
    
    loss_b = self.discriminator(batch)
    opt_b.zero_grad()
    self.manual_backward(loss_b)
    opt_b.step()
  ~~~

## Predict or Deploy

- model을 training 시킬 때, prediction을 위한 LightningModule의 3가지 방법(Option)이 있습니다.

### Option 1: sub-models

- Prediction을 위해 system 내의 model을 가져옵니다.

  ~~~python
  # ----------------------------------
  # to use as embedding extractor
  # ----------------------------------
  autoencoder = LitAutoEncoder.load_from_checkpoint('path/to/checkpoint_file.ckpt')
  encoder_model = autoencoder.encoder
  encoder_model.eval()

  # ----------------------------------
  # to use as image generator
  # ----------------------------------
  decoder_model = autoencoder.decoder
  decoder_model.eval()
  ~~~

### Option 2: Forward

- 원하는 대로 prediction을 진행할 수 있도록 **foward method**를 추가할 수 있습니다.
  ~~~python
  # ----------------------------------
  # using the AE to extract embeddings
  # ----------------------------------
  class LitAutoEncoder(LightningModule):
      def __init__(self):
          super().__init__()
          self.encoder = nn.Sequential()

      def forward(self, x):
          embedding = self.encoder(x)
          return embedding

  autoencoder = LitAutoEncoder()
  autoencoder = autoencoder(torch.rand(1, 28 * 28))
  ~~~
  
  ~~~python
  # ----------------------------------
  # or using the AE to generate images
  # ----------------------------------
  class LitAutoEncoder(LightningModule):
      def __init__(self):
          super().__init__()
          self.decoder = nn.Sequential()

      def forward(self):
          z = torch.rand(1, 3)
          image = self.decoder(z)
          image = image.view(1, 1, 28, 28)
          return image

  autoencoder = LitAutoEncoder()
  image_sample = autoencoder()
  ~~~
 
 ### Option 3 : Production
 
 - production systems에 대해서는, **onnx**나 **torchscript**가 훨씬 빠르게 동작합니다.
 - **forward method**를 추가하거나, **꼭 필요한 sub-model만을 trace**해야합니다.
  ~~~python
  # ----------------------------------
  # torchscript
  # ----------------------------------
  autoencoder = LitAutoEncoder()
  torch.jit.save(autoencoder.to_torchscript(), "model.pt")
  os.path.isfile("model.pt")
  ~~~
  
  ~~~python
  # ----------------------------------
  # onnx
  # ----------------------------------
  with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
      autoencoder = LitAutoEncoder()
      input_sample = torch.randn((1, 28 * 28))
      autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
      os.path.isfile(tmpfile.name)
  ~~~
  
## Using CPUs/GPUs/TPUs

- CPUs/GPUs/TPUs를 사용하기 위해서 code를 바꿀 필요가 없습니다.
- **Trainer**의 options를 바꾸는 것으로도 가능해집니다.
  ~~~python
  # train on CPU
  trainer = Trainer()
  ~~~
  
  ~~~python
  # train on 8 CPUs
  trainer = Trainer(num_processes=8)
  ~~~
  
  ~~~python
  # train on 1024 CPUs across 128 machines
  trainer = pl.Trainer(
      num_processes=8,
      num_nodes=128
  )
  ~~~
  
  ~~~python
  # train on 1 GPU
  trainer = pl.Trainer(gpus=1)
  ~~~
  
  ~~~python
  # train on multiple GPUs across nodes (32 gpus here)
  trainer = pl.Trainer(
      gpus=4,
      num_nodes=8
  )
  ~~~
  
  ~~~python
  # train on gpu 1, 3, 5 (3 gpus total)
  trainer = pl.Trainer(gpus=[1, 3, 5])
  ~~~
  
  ~~~python
  # Multi GPU with mixed precision
  trainer = pl.Trainer(gpus=2, precision=16)
  ~~~
  
  ~~~python
  # Train on TPUs
  trainer = pl.Trainer(tpu_cores=8)
  ~~~
  
## Checkpoints

- Lightning은 자동으로 model을 save합니다.
- 한번 train을 시키면, 아래와 같이 작성하여 checkpoint를 불러올 수 있습니다.
  ~~~python
  model = LitModel.load_from_checkpoint(path)
  ~~~

- 위의 checkpoint는 model을 init 하기 위한 모든 argument들을 포함하고 있으며, state dict를 설정합니다.
- 직접 구현하는 것을 원한다면, 아래와 같이 작성하면 됩니다.
  ~~~python
  # load the ckpt
  ckpt = torch.load('path/to/checkpoint.ckpt')

  # equivalent to the above
  model = LitModel()
  model.load_state_dict(ckpt['state_dict'])
  ~~~

 
## Data flow

- 각 loop (training, validation, test)에는 우리가 구현할 수 있는 3 가지 hooks들이 있습니다.
  - x_step
  - x_step_end
  - x_epoch_end

- data flow를 이해하기 위해서, training loop (ie: x=training)를 이용해보겠습니다.
  ~~~python
  out = []
  for batch in data:
    out = training_step(batch)
    outs.append(out)
  trainin_epoch_end(outs)
  ~~~
  
- 위의 코드에 대한 Lightning 코드는 아래와 같습니다.
  ~~~python
  def training_step(self, batch, batch_idx):
    prediction = ...
    return prediction
    
  def training_epoch_end(self, training_step_outputs):
    for prediction in predictions:
      # do something with these
  ~~~

- DP 혹은 DDP2의 분할 모드를 사용하는 경우(ie: GPU간 batch 분할), x_step_end를 사용하여 수동으로 aggregate(집계)하거나 이것을 자동으로 aggregate(집계)하도록 구현하지 마십시오.
  ~~~python
  for batch in data:
    model_copies = copy_model_per_gpu(model, num_gpus)
    batch_split = split_batch_per_gpu(batchm, num_gpus)
    
    gpu_outs = []
    for model, batch_part in zip(model_copies, batch_split):
      # LightniongModule hook
      gpu_out = model.training_step(batch_part)
      gpu_puts.append(gpu_out)
      
    # LightningModule hook
    out = training_step_end(gpu_outs)
  ~~~
  
- 이에 대한 Lightning 코드는 다음과 같습니다.
  ~~~python
  def training_step(self, batch, batch_idx):
    loss = ...
    return loss
  
  def training_step_end(self, losses):
    gpu_0_loss = losses[0]
    gpu_1_loss = losses[1]
    return (gpu_0_loss + gpu_1_loss) * 1/2
  ~~~

## Logging

- Tensorboard, 가장 선호하는 logger, 그리고 progress bar에 log를 전달하기 위해서, *log()* method를 사용합니다.
- *log()* 는 LightningModule의 어떤 method로부터든 호출될 수 있습니다.
  ~~~python
  def training_step(self, batch, batch_idx):
    self. log('my_metric', x)
  ~~~

- *log()*  method는 몇가지 mode option을 가집니다. (T/F)
  - **on_step** : training의 step에서 metric의 log를 남깁니다.
  - **on_epoch** : epoch의 끝에서 자동으로 축적하며, log를 남깁니다.
  - **prog_bar** : progress bar에 log를 남깁니다.
  - **logger** : Tensorboard와 같은 logger에 logs를 전달합니다.

- log의 호출 위치에 따라 Lightning은 사용자에게 적합한 option를 자동으로 결정합니다. 그러나 수동으로 플래그를 설정하여 기본 동작을 재정의할 수 있습니다.
 ~~~python
 def training_step(self, batch, batch_idx):
  self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=Ture)
 ~~~

- 사용하는 logger의 어떤 method든 사용할 수 있습니다.
  ~~~python
  def training_step(self, batch, batch_idx):
    tensorboard = self.logger.experiment
    tensorboard.any_summary_writer_method_you_want())
  ~~~
  
- 한번 training이 시작되면, 선호하는 logger를 사용하거나, Tensorboard logs를 booting핟여 logs를 보여줍니다.
  ~~~python
  tensorboard --logdir ./lightning_logs
  ~~~

## optional extensions

### Callbacks

- callback은 Training loog의 임의의 부분에서 실행할 수 있는 임의 self-contained program입니다.
- 다음은 not-so-fancy learning rate decay rule을 추가하는 예입니다.
  ~~~python
  from pytorch_lightning.callbacks import Callback

  class DecayLearningRate(Callback):

      def __init__(self):
          self.old_lrs = []

      def on_train_start(self, trainer, pl_module):
          # track the initial learning rates
          for opt_idx, optimizer in enumerate(trainer.optimizers):
              group = [param_group['lr'] for param_group in optimizer.param_groups]
              self.old_lrs.append(group)

      def on_train_epoch_end(self, trainer, pl_module, outputs):
          for opt_idx, optimizer in enumerate(trainer.optimizers):
              old_lr_group = self.old_lrs[opt_idx]
              new_lr_group = []
              for p_idx, param_group in enumerate(optimizer.param_groups):
                  old_lr = old_lr_group[p_idx]
                  new_lr = old_lr * 0.98
                  new_lr_group.append(new_lr)
                  param_group['lr'] = new_lr
              self.old_lrs[opt_idx] = new_lr_group

  # And pass the callback to the Trainer
  decay_callback = DecayLearningRate()
  trainer = Trainer(callbacks=[decay_callback])
  ~~~

### LightningDataModules

- DataLoaders와 data processing code는 결국 흩어져 버리는 경향이 있습니다.
- data code를 LightningDataModule로 구성하여 재사용할 수 있습니다.
  ~~~python
  class MNISTDataModule(LightningDataModule):

      def __init__(self, batch_size=32):
          super().__init__()
          self.batch_size = batch_size

      # When doing distributed training, Datamodules have two optional arguments for
      # granular control over download/prepare/splitting data:

      # OPTIONAL, called only on 1 GPU/machine
      def prepare_data(self):
          MNIST(os.getcwd(), train=True, download=True)
          MNIST(os.getcwd(), train=False, download=True)

      # OPTIONAL, called for every GPU/machine (assigning state is OK)
      def setup(self, stage: Optional[str] = None):
          # transforms
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])
          # split dataset
          if stage == 'fit':
              mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
              self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
          if stage == 'test':
              self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

      # return the dataloader for each split
      def train_dataloader(self):
          mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
          return mnist_train

      def val_dataloader(self):
          mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
          return mnist_val

      def test_dataloader(self):
          mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
          return mnist_test
  ~~~
  
- Lightning Data Module은 여러 프로젝트에서 데이터 분할 및 변환을 공유하고 재사용할 수 있도록 설계되었습니다. 다운로드, 토큰화, 처리 등 데이터 처리에 필요한 모든 단계를 캡슐화합니다.
- 이제 LightningdataModule을 Trainer에게 간단히 전달할 수 있습니다.
  ~~~python
  # init model
  model = LitModel()

  # init data
  dm = MNISTDataModule()

  # train
  trainer = pl.Trainer()
  trainer.fit(model, dm)

  # test
  trainer.test(datamodule=dm)
  ~~~
  
