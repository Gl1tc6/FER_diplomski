import torch
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pt_deep as ptd
import ksvm_wrap as ksvm
import data as d
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, l2_reg=0.01):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.l2_reg = l2_reg  # Jačina L2 regularizacije

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Ravnanje slike u vektor
        return torch.softmax(self.linear(x), dim=1)

    def get_weights(self):
        return self.linear.weight.data  # Težine: (10, 784)

def train_linear_model(X, y_train, l2_reg=0.01):
    print("Training Linear model!")
    model = LinearModel(l2_reg=l2_reg)
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=l2_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=l2_reg)
    
    X_flat = X.view(-1, 28*28) 
    
    for iter in range(1000):
        outputs = model(X_flat)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % 100 == 0:
            print(f"Iter {iter}: loss = {loss.item():.4f}")
    
    # Vizualizacija s normalizacijom
    weights = model.get_weights()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        img = weights[i].view(28, 28).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Min-max skaliranje
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Klasa {i}')
        ax.axis('off')
    plt.suptitle(f'Težine s L2={l2_reg}')
    plt.show()
    
def visualize_deep(model, title="PTDeep Effective Weights"):
    with torch.no_grad():
        if len(model.weights) == 1:
            # Single-layer model: direct weights
            effective_weights = model.weights[0]
        else:
            # Multi-layer model: multiply all weights (ignoring activations)
            effective_weights = model.weights[0].clone()
            for w in model.weights[1:-1]:
                effective_weights = torch.mm(effective_weights, w)
            effective_weights = torch.mm(effective_weights, model.weights[-1])
    
    # Normalize and plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        img = effective_weights[:, i].reshape(28, 28).cpu().detach().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f'Klasa {i}')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def visualize_ksvm(model, X_train, y_train, title="KSVM Class Prototypes"):
    support_vectors = X_train[model.support()]
    support_labels = y_train[model.support()] 
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        # Odaberi potporne vektore za klasu i
        mask = (support_labels == i)
        if mask.sum() == 0:
            img = np.zeros((28, 28))  # Prazna slika ako nema potpornih vektora
        else:
            prototype = np.mean(support_vectors[mask], axis=0)  # Prosjek potpornih vektora
            prototype = prototype.reshape(28, 28)
            prototype = (prototype - prototype.min()) / (prototype.max() - prototype.min())  # Normalizacija
            img = prototype
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Klasa {i}')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()


dataset_root = '/tmp/mnist'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

# Linear model
train_linear_model(x_train, y_train, l2_reg=0.001)

# Deep nn
# dim_config = [
#     [784, 10],
#     [784, 100, 10]
#     #[784, 100, 100, 10]
# ]

# Yoh_train = d.class_to_onehot(y_train.numpy())  # Pretvori y_train u one-hot
# Yoh_train = torch.Tensor(Yoh_train)  # Pretvori u torch.Tensor

# for dims in dim_config:
#     model = ptd.PTDeep(dims, torch.relu)
#     ptd.train(
#         model, 
#         x_train.view(-1, 784),  # Ravnanje ulaza (N, 784)
#         Yoh_train,              # One-hot enkodirani y_train
#         param_niter=1000, 
#         param_delta=0.1, 
#         param_lambda=0.0001
#     )

#     X_test_flat = x_test.view(-1, 784).numpy()
#     probs = ptd.eval(model, X_test_flat)
#     Y_pred = np.argmax(probs, axis=1)
#     acc, prec_rec, _ = d.eval_perf_multi(Y_pred, y_test.numpy())
#     print(f"PTDeep {dims} Test Accuracy: {acc:.4f}\n")
#     for c, (prec, rec) in enumerate(prec_rec):
#         print(f"Class {c}\nPrecision:\t{prec}\nRecall:\t{rec}\n\n")
    #visualize_deep(model)
    
# # SVM model
# X_train_flat = x_train.view(x_train.shape[0], -1).numpy()
# model = ksvm.KSVMWrap(X_train_flat, y_train, param_svm_c=10)
# preds = model.predict(X_train_flat)
# acc, prec_rec, _ = d.eval_perf_multi(preds, y_train)
    
# print(f'Accuracy: {acc}\n')
# for c, (prec, rec) in enumerate(prec_rec):
#     print(f"Class {c}\nPrecision:\t{prec}\nRecall:\t{rec}\n")
# visualize_ksvm(model, X_train_flat, y_train, title="KSVM Prototipovi (C=10)")