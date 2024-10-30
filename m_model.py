import torch
from torch import nn, Tensor

from m_unet import UNet
from m_tools import create_sin_cos_encoding

class Model(nn.Module):
  def __init__(self,
    in_ch:int=3, out_ch:int=3, w_size:int=16, h_size:int=16,
    steps:int=1000, categories:int=5, embedding:int=256,
    device='cpu', dtype=torch.float32
  ):
    factory_kwargs = {
      'device': device,
      'dtype': dtype,
    }
    super().__init__()

    w_space = torch.arange(start=0, end=w_size, requires_grad=False)
    h_space = torch.arange(start=w_size, end=w_size+h_size, requires_grad=False)
    self.register_buffer('w_space', w_space)
    self.register_buffer('h_space', h_space)

    w_weight = create_sin_cos_encoding(embedding, w_size)
    h_weight = create_sin_cos_encoding(embedding, h_size)
    weight = torch.cat((w_weight, h_weight), dim=0).to(device)

    self.space_emb = nn.Embedding(num_embeddings=w_size + h_size, embedding_dim=embedding, _weight=weight, **factory_kwargs)
    self.category_emb = nn.Embedding(categories, embedding, **factory_kwargs)
    self.step_emb = nn.Embedding(steps, embedding, _weight=create_sin_cos_encoding(embedding, steps).to(device), **factory_kwargs)

    self.unet = UNet(in_ch, out_ch, embedding, **factory_kwargs)

  def __call__(self, X:Tensor, t:Tensor, c:Tensor=None) -> Tensor:
    return super().__call__(X, t, c)

  def forward(self, X:Tensor, t:Tensor, c:Tensor) -> Tensor:
    w_space, h_space = self.get_buffer('w_space'), self.get_buffer('h_space')

    t = self.step_emb(t)
    s = self.space_emb(w_space), self.space_emb(h_space)
    c = self.category_emb(c) if c is not None else None

    X = self.unet(X, t, s, c)

    return X