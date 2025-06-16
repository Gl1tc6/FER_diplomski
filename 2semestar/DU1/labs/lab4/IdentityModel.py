import torch
import torch.nn as nn

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        batch_size = img.size(0)
        feats = img.view(batch_size, -1)
        return feats