import torch
from torch import nn, Tensor
import torch.nn.functional as F

from m_convolutional import Convolutional

class DownBlock(nn.Module):
  def __init__(self,
    in_ch:int, out_ch:int, step_embedding:int=512, category_embedding:int=512,
    active_fn=F.silu,
    device='cpu', dtype=torch.float32
  ) -> None:
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super().__init__()

    self.active_fn = active_fn

    self.conv = nn.Sequential(
      nn.MaxPool2d(2),
      Convolutional(in_ch, in_ch, residual=True, active_fn=active_fn, **factory_kwargs),
      Convolutional(in_ch, out_ch, residual=False, active_fn=active_fn, **factory_kwargs),
    )

    self.step_emb = nn.Linear(step_embedding, out_ch, **factory_kwargs)
    self.category_emb = nn.Linear(category_embedding, out_ch // 2, **factory_kwargs)

  def __call__(self, X:Tensor, step:Tensor, categories:Tensor=None) -> Tensor:
    return super().__call__(X, step, categories)

  def forward(self, X:Tensor, step:Tensor, categories:Tensor=None) -> Tensor:
    X = self.conv(X)

    if categories is not None:
      category_emb = self.category_emb(self.active_fn(categories))[:, :, None, None]\
      .repeat(1, 1, X.shape[-2], X.shape[-1])
      X[:, ::2, :, :] += category_emb

    step_emb = self.step_emb(self.active_fn(step))[:, :, None, None]\
      .repeat(1, 1, X.shape[-2], X.shape[-1])

    return X + step_emb

if __name__ == '__main__':
  from m_tools import create_sin_cos_encoding

  DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  CATEGORIES = 10
  EMBEDDING = 256
  STEPS = 1000
  BATCH = 32

  factory_kwargs = {
    'device': DEVICE,
    'dtype': torch.float32,
  }

  category_emb = nn.Embedding(CATEGORIES, EMBEDDING, **factory_kwargs)
  step_emb = nn.Embedding(STEPS, EMBEDDING, _weight=create_sin_cos_encoding(EMBEDDING, STEPS), **factory_kwargs)

  down = DownBlock(3, 64, EMBEDDING, EMBEDDING)

  X = torch.randn(BATCH, 3, 64, 64, **factory_kwargs)
  t = torch.randint(low=0, high=STEPS, size=(BATCH,), dtype=torch.long, device=DEVICE)
  categories = torch.randint(low=0, high=CATEGORIES, size=(BATCH,), dtype=torch.long, device=DEVICE)

  t = step_emb(t)
  categories = category_emb(categories)
  Z = down(X, t, categories)

  print('t', t.shape)
  print('c', categories.shape)
  print('X', X.shape)
  print('Z', Z.shape)
