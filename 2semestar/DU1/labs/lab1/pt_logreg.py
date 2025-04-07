import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data as d
import matplotlib.pyplot as plt

class PTLogreg(nn.Module):
  def __init__(self, D, C):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """

    # inicijalizirati parametre (koristite nn.Parameter):
    # imena mogu biti self.W, self.b
    # ...
    super(PTLogreg, self).__init__()
    self.W = nn.Parameter(torch.randn(D, C), requires_grad=True)
    self.b = nn.Parameter(torch.randn(C), requires_grad=True)

  def forward(self, X):
    # unaprijedni prolaz modela: izračunati vjerojatnosti
    #   koristiti: torch.mm, torch.softmax
    # ...
    s = torch.mm(X, self.W) + self.b
    return torch.softmax(s, dim=1)

  def get_loss(self, X, Yoh_):
    # formulacija gubitka
    #   koristiti: torch.log, torch.exp, torch.sum
    #   pripaziti na numerički preljev i podljev
    # ...
    p = self.forward(X)
    return - torch.sum(Yoh_ * torch.log(p), dim=1).mean()


def train(model:PTLogreg, X, Yoh_, param_niter, param_delta, param_lambda):
  """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
     - param_lambda: regularization
  """
  model.train()
  
  # inicijalizacija optimizatora
  # ...
  opt = optim.SGD(model.parameters(), lr=param_delta)

  # petlja učenja
  # ispisujte gubitak tijekom učenja
  # ...
  for i in range(1, int(param_niter)+1):
      mod_loss = model.get_loss(X, Yoh_) + param_lambda*torch.norm(model.W, 2)
      mod_loss.backward()
      if i % (param_niter // 10) == 0:
            print(f"Finished {int((i / param_niter) * 100)}% of iterations ({i}/{int(param_niter)})")
            print(f"[CUR LOSS]:\t{mod_loss}\n")
            
      opt.step()
      opt.zero_grad()
  return


def eval(model:PTLogreg, X:np.array):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  # ulaz je potrebno pretvoriti u torch.Tensor
  # izlaze je potrebno pretvoriti u numpy.array
  # koristite torch.Tensor.detach() i torch.Tensor.numpy()
  model.eval()
  return model.forward(torch.Tensor(X)).detach().numpy()

def pt_logreg_decfun(model):
    return lambda X: np.argmax(eval(model, X), axis=1)
#pt_logreg_decfun = lambda model: (lambda X: np.argmax(eval(model, X), axis=1))

if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)

  # instanciraj podatke X i labele Yoh_
  X, Y = d.sample_gauss_2d(3, 500)
  Yoh_ = d.class_to_onehot(Y)
  
  X = torch.Tensor(X)
  Yoh_ = torch.Tensor(Yoh_)
  
  X = (X-X.mean(dim=0)) / X.std(dim=0) # Izbjeljivanje ne radi :(

  # definiraj model:
  ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

  # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
  train(ptlr, X, Yoh_, 2000, 0.1, 0.001)

  # dohvati vjerojatnosti na skupu za učenje
  probs = eval(ptlr, X)

  # ispiši performansu (preciznost i odziv po razredima)
  Y_ = np.argmax(probs, axis=1)
  acc, prec_rec, conf_mtrx = d.eval_perf_multi(Y_, Y)
  print(f'Accuracy: {acc}\n')
  for c, (prec, rec) in enumerate(prec_rec):
      print(f"Class {c}\nPrecision:\t{prec}\nRecall:\t{rec}\n")

  # iscrtaj rezultate, decizijsku plohu
  X_ = X.detach().numpy()
  decfun = pt_logreg_decfun(ptlr)
  rect=(np.min(X_, axis=0), np.max(X_, axis=0))
  d.graph_surface(decfun, rect, offset=0.5)
  d.graph_data(X_, Y_, Y)
  plt.show()