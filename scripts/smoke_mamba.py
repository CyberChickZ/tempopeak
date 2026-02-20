import torch
from mamba_ssm import Mamba

B, T, D = 2, 64, 128

model = Mamba(d_model=D).cuda()
x = torch.randn(B, T, D).cuda()

y = model(x)

print("input shape:", x.shape)
print("output shape:", y.shape)
