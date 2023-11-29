import torch
import torch.nn as nn
import json

class LLM(nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super(LLM, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_size,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=num_classes)
        )
        
    def forward(self, x):
        return self.layer_stack(x)