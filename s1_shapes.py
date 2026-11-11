import torch
import torch.nn as nn

x = torch.randn(2, 1, 28, 28)          # (batch, channel, H, W)
print("input  :", x.shape)

conv1 = nn.Conv2d(1, 10, kernel_size=5)
pool = nn.MaxPool2d(2)

y = conv1(x)
print("conv1  :", y.shape)
y = pool(y)
print("pool   :", y.shape)

print("conv1.weight:", conv1.weight.shape)
print("conv1.bias  :", conv1.bias.shape)
n = sum(p.numel() for p in conv1.parameters())
print("conv1 params:", n)

same = nn.Conv2d(1, 10, kernel_size=5, padding=2)
print("padding=2   :", same(x).shape)