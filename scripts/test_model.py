import torch
from models.model import EventModel

B, T, H, W = 2, 64, 224, 224

model = EventModel().cuda()

x = torch.randn(B, T, 3, H, W).cuda()

p_logits = model(x)
p = torch.softmax(p_logits, dim=1)

print("output shape:", p_logits.shape)
print("sum over time (after softmax):", p.sum(dim=1))
