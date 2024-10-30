import torch
from torch import Tensor

def linear_shedule(num_steps=1000, start=1e-4, end=2e-2) -> Tensor:
  return torch.linspace(start, end, num_steps)

def cosine_schedule(num_steps, s=8e-3):
  def f(t:Tensor) -> Tensor:
    return torch.cos((t / num_steps + s) / (1 + s) * 0.5  * torch.pi) ** 2

  x = torch.linspace(0, num_steps, num_steps+1)
  alphas_cumprod = f(x) / f(torch.tensor([0]))
  beta_cosine = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
  return torch.clip(beta_cosine, 0.0001, 0.999)

class Diffusion:
  def __init__(self, steps:int, beta:Tensor, device='cpu'):
    self.steps = steps
    self.device = device
    self.beta = beta
    self.alpha = 1.0 - self.beta
    self.alpha_hat = torch.cumprod(self.alpha, dim=0)

  def noise_image(self, X:Tensor, t:Tensor) -> tuple[Tensor, Tensor]:
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
    sqrt_one_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
    e = torch.randn_like(X, device=self.device)
    return sqrt_alpha_hat * X + sqrt_one_alpha_hat * e, e

  def sample_timesteps(self, n: int) -> Tensor:
    return torch.randint(low=1, high=self.steps, size=(n,), device=self.device)