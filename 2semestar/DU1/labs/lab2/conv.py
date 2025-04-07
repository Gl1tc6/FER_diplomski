import torch
from torch import nn
from torch.utils.data import DataLoader
 
class CovolutionalModel(nn.Module):
  def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.conv2 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    # potpuno povezani slojevi
    self.fc1 = nn.Linear(..., fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)
    
    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.maxpool(h)
    h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    h = self.conv2(h)
    h = torch.relu(self.maxpool(h))
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return logits