import torch
import torchvision
import numpy as np

class MNIST():
    def __init__(self, image_size: int = 64, batch_size: int = 64) -> None:
        self.image_size = image_size
        self.batch_size = batch_size

        self.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(self.image_size),
        torchvision.transforms.ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
        ])
    
    def reverse_transform(self):
        reverse_transform = 0,
        return reverse_transform
    
    
    def dataset(self):    
        self.train_dataset = torchvision.datasets.MNIST(
            root="datasets/MNIST",
            download=True,
            train=True,
            transform=self.transform)
        
        self.test_dataset = torchvision.datasets.MNIST(
            root="datasets/MNIST",
            download=True,
            train=False,
            transform=self.transform)
        
        return self.train_dataset, self.test_dataset
    
    def dataloader(self):
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True)
        
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,)
        
        return self.train_dataloader, self.test_dataloader


