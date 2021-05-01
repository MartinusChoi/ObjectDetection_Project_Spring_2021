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

 
 
 
 
