import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ..util

def train(model_name: str,
          epochs: int, 
          batch_size: int,
          optim_name: str,
          imagenet_path: str,
          learning_rate: float = 0.1,
          shuffle: bool = True,
          num_workers: int = 1,
          include_val: bool = False,
          seed: int = None) -> None:
    model = get_model(model_name, 'pytorch')
    optimizer = get_optim(optim_name, 'pytorch')
    criterion = get_loss(loss_name, 'pytorch')
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        
    cudnn.benchmark = True # Let cuDNN find the optimized algorithm for the model
    
    train_dir = os.path.join(imagenet_path, 'train')
    val_dir = os.path.join(imagenet_path, 'val')
    if (not os.path.exists(train_dir) or not os.path.exists(val_dir)):
        raise FileNotFoundError("Dataset directory should contain `train` and `val` folders.")
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataset = datasets.ImageFolder(
        val_dataset,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
        
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if include_val:
            _validation(val_loader, model, criterion)
            
def _validation(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            loss = criterion(output, labels)
            