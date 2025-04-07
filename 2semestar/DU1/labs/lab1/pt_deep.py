import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data as d
import matplotlib.pyplot as plt

class PTDeep(nn.Module):
    def __init__(self, nn_dims, act_fun):
        super(PTDeep, self).__init__()
        self.n_layers = len(nn_dims) -1
        self.act_fun = act_fun
        
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
    
        for i in range(self.n_layers):
            weight = nn.Parameter(torch.randn(nn_dims[i], nn_dims[i+1]))
            bias = nn.Parameter(torch.zeros(nn_dims[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
    def forward(self, X):
        for i in range(len(self.biases) - 1):
            X = torch.mm(X, self.weights[i]) + self.biases[i]
            X = self.act_fun(X)
        
        X = torch.mm(X, self.weights[-1]) + self.biases[-1]
        return torch.softmax(X, dim=1)
    
    def get_loss(self, X, Yoh_, param_lambda=0):
        # Get predicted probabilities
        p = self.forward(X)
        # Calculate cross-entropy loss (with small epsilon for numerical stability)
        return -torch.sum(Yoh_ * torch.log(p), dim=1).mean()
    

    def count_params(self):
        sum_params = 0
        for p in self.named_parameters():
            print(f'Layer: {p[0]}, dims: {p[1].shape}')
            sum_params += p[1].numel()
        print(f'Total number of parameters: {sum_params}')
        
def train(model:PTDeep, X, Yoh_, param_niter:int, param_delta:float, param_lambda:float):
    """Arguments:
     - train_data: model inputs [NxD], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
     - param_lambda: regularization
    """
    model.train()
    opt = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    for i in range(int(param_niter)):
        mod_loss = model.get_loss(X, Yoh_)
        mod_loss.backward()
        if i % (param_niter // 10) == 0:
            print(f"Finished {int((i / param_niter) * 100)}% of iterations ({i}/{int(param_niter)})")
            print(f"[CUR LOSS]:\t{mod_loss}\n")
        opt.step()
        opt.zero_grad()
    return

def eval(model:PTDeep, X:np.array):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  model.eval()
  return model.forward(torch.Tensor(X)).detach().numpy()

def pt_deep_decfun(model):
    return lambda X: np.argmax(eval(model, X), axis=1)

if __name__ == "__main__":
    np.random.seed(100)

    X, Y = d.sample_gmm_2d(6, 2, 10)
    Yoh_ = d.class_to_onehot(Y)
    
    X = torch.Tensor(X)
    Yoh = torch.Tensor(Yoh_)
    
    X = (X - X.mean(dim=0)) / X.std(dim=0) # nikako drukčije loss spustiti - Izbjeljivanje
    
    model = PTDeep([2, 10, 10, 2], torch.relu)
    #model = PTDeep([2, 10, 2], torch.sigmoid)
    #model = PTDeep([2, 2], torch.relu)

    train(model, X, Yoh, 10000, 0.1, 0.0001)
    print(model.get_loss(X, Yoh))
    
    probs = eval(model, X)

    Y_pred = np.argmax(probs, axis=1)
    acc, prec_rec, conf_mtrx = d.eval_perf_multi(Y_pred, Y)
    
    print(f'Accuracy: {acc}\n')
    for c, (prec, rec) in enumerate(prec_rec):
        print(f"Class {c}\nPrecision:\t{prec}\nRecall:\t{rec}\n")

    # Plot decision surface and data points
    decfun = pt_deep_decfun(model)
    rect = (np.min(X.numpy(), axis=0), np.max(X.numpy(), axis=0))
    d.graph_surface(decfun, rect, offset=0)
    d.graph_data(X.numpy(), Y, Y_pred)
    plt.show()
    model.count_params() 