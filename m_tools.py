import torch, math
from torch import Tensor

def create_sin_cos_encoding(d_model:int, max_length:int, dtype=torch.float32, device='cpu') -> Tensor:
  factory_kwargs = {
    'dtype': dtype,
    'device': device,
  }

  pe = torch.zeros(max_length, d_model, **factory_kwargs)
  position = torch.arange(0, max_length, **factory_kwargs).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d_model, 2, **factory_kwargs) * -(math.log(1e+4) / d_model))

  pe[:, 0::2] = (position * div_term).sin()
  pe[:, 1::2] = (position * div_term).cos()

  return pe

def batch_image_split(batch:Tensor, splits:int) -> Tensor:
  sw = batch.shape[-2] // splits
  sh = batch.shape[-1] // splits

  return torch.stack([
    batch[:, :, j*sw:(j+1)*sw, i*sh:(i+1)*sh]
      for i in range(splits)
      for j in range(splits)
  ], dim=1).view(-1, batch.shape[1], sw, sh)

def batch_image_join(batch:Tensor, splits:int) -> Tensor:
  X = batch.view(-1, splits, splits, batch.shape[1], batch.shape[2], batch.shape[3])

  X = torch.cat([X[:, i] for i in range(splits)], dim=-1)
  X = torch.cat([X[:, i] for i in range(splits)], dim=-2)

  return X

def batch_image_to_batch_sequential(X:Tensor) -> Tensor:
  sw, sh = X.shape[-2], X.shape[-1]
  return X.view(X.shape[0], X.shape[1], -1).swapaxes(-1, 1), sw, sh

def batch_sequential_to_batch_image(X:Tensor, sw:int, sh:int) -> Tensor:
  return X.swapaxes(-1, 1).view(X.shape[0], X.shape[-1], sw, sh)