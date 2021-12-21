import os
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        if config.model_body == 'resnet18':
            self.body = torchvision.models.resnet18()
        elif config.model_body == 'resnet50':
            self.body = torchvision.models.resnet50()
        elif config.model_body == 'resnet152':
            self.body = torchvision.models.resnet152()
        elif config.model_body == 'efficientNet':
            self.body = torchvision.models.efficientnet_b2()

        if config["body_weights_file"] is not None:
            print("loading pretrained body weights", config["body_weights_file"])
            state_dict = torch.load(config["body_weights_file"])
            self.body.load_state_dict(state_dict)

        hidden_size = config.hidden_size
        self.net = nn.Sequential(

            self.body,
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            # nn.Dropout(p=config.dropout),
            # nn.Linear(1000, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(),
            # nn.Dropout(p=config.dropout),
            nn.Linear(1000, config.n_classes),
            nn.LogSoftmax(dim=1),
            
        )

    def forward(self, x):
        return self.net(x)
