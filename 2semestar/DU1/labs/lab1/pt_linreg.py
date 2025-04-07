import torch
import torch.nn as nn
import torch.optim as optim
import random

## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

len = 5
# Proizvoljni broj točakan
X = torch.randn(len).clone().detach()
Y = (a * X + b + 0.5*torch.randn(len)).clone().detach()

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

runs = 100
for i in range(runs+1):
    Y_ = a*X + b

    dif = (Y-Y_)
    model_loss = torch.mean(dif**2)
    model_loss.backward() 
    if i % (runs // 10) == 0:
        print(f"Finished {int((i / runs) * 100)}% of iterations ({i}/{runs})")
        grd_a = torch.mean(-2 * dif * X)
        grd_b = torch.mean(-2 * dif)
        print("Analitički")
        print(f'a: {grd_a}, b: {grd_b}')
        print("PyTorch")
        print(f'a: {float(a.grad)}, b: {float(b.grad)}\n')
        

    optimizer.step()
    optimizer.zero_grad()