
import torch
from torch.autograd import Variable
import numpy as np

x = Variable(torch.Tensor([[1,2,3],[4,5,6]]),requires_grad=True)
y = Variable(torch.Tensor([[1, 2, 3], [4, 5, 6]]), requires_grad=True)

size=Variable(torch.Tensor([2]),requires_grad=True)

m = x*size
n = m*size
L=n.mean()
L.backward()
print(m.is_leaf,x.is_leaf)
print(m.grad,x.grad)