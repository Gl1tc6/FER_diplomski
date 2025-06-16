import sys
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torch.optim
from IdentityModel import IdentityModel
from task1 import MNISTMetricDataset
from torch.utils.data import DataLoader
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.Conv2d(num_maps_in, num_maps_out, k, padding=k//2, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.conv1 = _BNReluConv(input_channels, emb_size, k=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = _BNReluConv(emb_size, emb_size, k=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = _BNReluConv(emb_size, emb_size, k=3)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = self.conv1(img)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        
        x = self.global_avg_pool(x)
        
        x = x.squeeze()
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        p = 2
        eps = 1e-6
        fst = torch.linalg.norm(a_x - p_x, ord=p, dim=1) + eps
        scnd = torch.linalg.norm(a_x - n_x, ord=p, dim=1) + eps

        loss = F.relu(fst-scnd+1).mean()
        return loss
    

if __name__ == '__main__':
    
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "/tmp/mnist"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    #model = SimpleMetricEmbedding(1, emb_size).to(device)
    emb_size = 32
    model = IdentityModel().to(device) if sys.argv[1] == "id" else SimpleMetricEmbedding(1, emb_size).to(device)

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True if isinstance(model, SimpleMetricEmbedding) else False,
        pin_memory=False,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1
    )

    os.makedirs("saved", exist_ok=True)
    typeof = "SME" if isinstance(model, SimpleMetricEmbedding) else "Identity"
    model_path = os.path.join("saved", f"{typeof}_{time.time()}.model")

    emb_size = 32 if isinstance(model, SimpleMetricEmbedding) else 28*28
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    ) if isinstance(model, SimpleMetricEmbedding) else None

    if isinstance(model, SimpleMetricEmbedding):
        epochs = 3
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            t0 = time.time_ns()
            train_loss = train(model, optimizer, train_loader, device)
            print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
            if EVAL_ON_TEST or EVAL_ON_TRAIN:
                print("Computing mean representations for evaluation...")
                representations = compute_representations(model, train_loader, num_classes, emb_size, device)
            if EVAL_ON_TRAIN:
                print("Evaluating on training set...")
                acc1 = evaluate(model, representations, traineval_loader, device)
                print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
            if EVAL_ON_TEST:
                print("Evaluating on test set...")
                acc1 = evaluate(model, representations, test_loader, device)
                print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
            t1 = time.time_ns()
            print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}\n")
    elif isinstance(model, IdentityModel):
        representations = compute_representations(model, train_loader, num_classes, emb_size, device)
        acc1 = evaluate(model, representations, test_loader, device)
        print(f"Identity Model Accuracy: {acc1 * 100:.2f}%")
    
    torch.save({
        'epoch': epochs if isinstance(model, SimpleMetricEmbedding) else 0,
        'model_state':model.state_dict(),
        'optimizer_state': optimizer.state_dict() if isinstance(model, SimpleMetricEmbedding) else "None",
        'Acc': acc1,
        'emb_size':emb_size,
    }, model_path)