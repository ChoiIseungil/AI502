import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        
        layers = []
        in_dim = 3*32*32
        for dim in [1024, 512, 256, 128]:
            out_dim = dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(out_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, img):
        img = img.reshape(img.size(0), -1)
        logits = self.logit(img)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    def logit(self, img):
        return self.network(img.view(len(img), -1))
    