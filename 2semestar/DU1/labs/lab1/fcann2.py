import data
import numpy as np
import matplotlib.pyplot as plt


relu = lambda x: np.maximum(0,x)

def softmax(X):
    exps = np.exp(X - np.max(X, axis=1).reshape(-1,1))
    return exps / exps.sum(axis=1).reshape(-1,1)


class fcann2:
    def __init__(self, in_dim, out_dim, h_dim):
        scale_lay1 = 1/np.mean([in_dim, h_dim])
        self.lay1 = np.random.normal(scale=scale_lay1, size=(in_dim, h_dim))
        self.b1 = np.zeros(h_dim)
        
        scale_lay2 = 1/np.mean([h_dim, out_dim])
        self.lay2 = np.random.normal(scale=scale_lay2, size=(h_dim, out_dim))
        self.b2 = np.zeros(out_dim)
    
    def forward(self, X):
        self.s1 = np.dot(X, self.lay1) + self.b1
        self.h1 = relu(self.s1)
        self.s2 = np.dot(self.h1, self.lay2) + self.b2
        self.p = softmax(self.s2)
        return self.p
    

def fcann2_train(X, Y, param_niter:int, param_delta:float):
    hidden_dim = 5
    n, d = X.shape
    c = max(Y) + 1
    Y_ = data.class_to_onehot(Y)
    mod = fcann2(d, c, hidden_dim)
    
    for i in range(int(param_niter)):
        if i % (param_niter // 10) == 0:
            print(f"Finished {int((i / param_niter) * 100)}% of iterations ({i}/{int(param_niter)})")
        
        # Forward pass
        P = mod.forward(X)
        
        grad_s2 = P - Y_
        
        # bp second layer
        grad_lay2 = np.dot(mod.h1.T, grad_s2).mean()
        grad_b2 = np.sum(grad_s2, axis=0).mean()
        
        # bp first layer
        grad_h1 = np.dot(grad_s2, mod.lay2.T)
        grad_s1 = grad_h1 * (mod.s1 > 0)  # Derivative of ReLU
        grad_lay1 = np.dot(X.T, grad_s1) / n # .mean ne radi?
        grad_b1 = np.sum(grad_s1, axis=0) / n
        

        mod.lay1 -= param_delta * grad_lay1
        mod.b1 -= param_delta * grad_b1
        mod.lay2 -= param_delta * grad_lay2
        mod.b2 -= param_delta * grad_b2
    return mod


def fcann2_classify(X, model:fcann2):
    fw_pass = model.forward(X)
    return np.argmax(fw_pass, axis=1)
    

def fcann2_decfun(model):
    return lambda x: fcann2_classify(x, model)

if __name__=="__main__":
    np.random.seed(100)
  
    # get data
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    model = fcann2_train(X, Y_, 1e5, .05)
    Y = fcann2_classify(X, model)
    
    decfun = fcann2_decfun(model)
  
    # graph the data points
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, rect, offset=0)
    data.graph_data(X, Y_, Y, special=[])

    plt.show()

