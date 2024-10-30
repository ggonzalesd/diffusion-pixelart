import torch
from torch import nn, Tensor

from m_convolutional import Convolutional
from m_downblock import DownBlock
from m_attention import Attention
from m_upblock import UpBlock

class UNet(nn.Module):
  def __init__(self, in_ch:int, out_ch:int, embedding:int=256, device='cpu', dtype=torch.float32):
    factory_kwargs = {
      'device': device,
      'dtype': dtype,
    }
    super().__init__()

    self.inc1 = Convolutional(in_ch=in_ch, out_ch=64, **factory_kwargs)

    self.down1 = DownBlock(in_ch=64, out_ch=128, step_embedding=embedding, category_embedding=embedding, **factory_kwargs)
    self.sa1 = Attention(d_model=128, heads=8, layers=1, splits=2, spacing=2, size_emb=embedding, **factory_kwargs)
    self.down2 = DownBlock(in_ch=128, out_ch=256, step_embedding=embedding, category_embedding=embedding, **factory_kwargs)
    self.sa2 = Attention(d_model=256, heads=8, layers=1, splits=1, spacing=4, size_emb=embedding, **factory_kwargs)

    self.bot1 = Convolutional(256, 512, **factory_kwargs)
    self.bot2 = Convolutional(512, 256, **factory_kwargs)

    self.up1 = UpBlock(in_ch=256 + 128, out_ch=128, step_embedding=embedding, category_embedding=embedding, **factory_kwargs)
    self.sa3 = Attention(d_model=128, heads=8, layers=1, splits=2, spacing=2, size_emb=embedding, **factory_kwargs)
    self.up2 = UpBlock(in_ch=128 + 64, out_ch=64, step_embedding=embedding, category_embedding=embedding, **factory_kwargs)
    self.sa4 = Attention(d_model=64, heads=8, layers=1, splits=4, spacing=1, size_emb=embedding, **factory_kwargs)
    self.out = nn.Conv2d(64, out_ch, kernel_size=1, **factory_kwargs)

  def __call__(self, X:Tensor, t:Tensor, s:tuple[Tensor, Tensor], c:Tensor=None) -> Tensor:
    return super().__call__(X, t, s, c)

  def forward(self, X:Tensor, t:Tensor, s:tuple[Tensor, Tensor], c:Tensor) -> Tensor:
    Z1 = self.inc1(X)
    Z2 = self.down1(Z1, t, c)
    Z2 = self.sa1(Z2, s)
    Z3 = self.down2(Z2, t, c)
    Z3 = self.sa2(Z3, s)

    Z3 = self.bot1(Z3)
    Z3 = self.bot2(Z3)

    X = self.up1(Z3, t, Z2, c)
    X = self.sa3(X, s)
    X = self.up2(X, t, Z1, c)
    X = self.sa4(X, s)

    X:Tensor = self.out(X)

    return X
