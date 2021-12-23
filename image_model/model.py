from torch import nn
import torch.nn.functional as F
import timm # PyTorch Image Models

def create_model(model_name):
    model = timm.create_model(model_name,pretrained=True)
    
    #let's update the pretarined model:
    for param in model.parameters():
        param.requires_grad=False

    model.classifier = nn.Sequential(
        nn.Linear(in_features=1792, out_features=625), #1792 is the orginal in_features
        nn.ReLU(), #ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=3), 
    )
    return model