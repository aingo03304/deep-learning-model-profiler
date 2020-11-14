import torch
import pyprof
from torchvision import models

def load_models(model_name):
    return models.