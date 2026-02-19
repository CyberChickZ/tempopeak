import torch
from models.model import EventModel

B, T, H, W = 2, 64, 224, 224

model = EventModel().cuda()

x = torch.randn(B, T, 3, H, W).cuda()

p = model(x)

print("output shape:", p.shape)
print("sum over time:", p.sum(dim=1))
