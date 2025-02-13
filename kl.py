import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

# 创建两个正态分布
p = Normal(0, 1)  # 均值为0，标准差为1
q = Normal(1, 1)  # 均值为1，标准差为1
print(p)
print(q)
# 计算KL散度
kl_div = kl_divergence(p, q)

print(kl_div)

