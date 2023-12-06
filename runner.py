from model import unet
from datasets import MNIST
from scheduler import ddpm
from model.helpers import num_to_groups
from utils.utils import dict2namespace

import torch
import numpy as np
import torchvision
from PIL import Image
import yaml

# Load Model Config
config_path = "config/MNIST.yaml"
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

config = dict2namespace(config)

# Initialize Dataset
MNIST_Data = MNIST.MNIST(batch_size=config.data.batch_size, image_size=config.data.image_size)
train_dataset, test_dataset = MNIST_Data.dataset()
train_dataloader, test_dataloader = MNIST_Data.dataloader()

# Initialize Model
model = unet.UNet(in_channels=1)
model = model.to(config.training.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

# Initialize Scheduler
scheduler = ddpm.DDPM(config) 
loss = []

# Training
for epoch in range(config.training.epochs):
   for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        batch_size = batch[0].shape[0]
        batch = batch[0].to(config.training.device)
        
        # Sample Radnom Timestemps
        t = torch.randint(0, config.scheduler.num_diffusion_timestemps, (config.data.batch_size,), device=config.training.device).long()

        # Calculate Loss
        loss = scheduler.p_losses(model, batch, t, loss_type="l2")

        # Update Model
        loss.backward()
        optimizer.step()

        if step % 18000 == 0:
            # Print Loss
            print("Loss:", loss.item())
            
            # Save Model and Image Samples at last Timestemp
            samples = scheduler.p_sample(model)
            torchvision.utils.save_image(samples[-1], f"saves/outputs/{config.data.dataset}_{step}_{epoch}.png")   
            torch.save(model.state_dict(), f"saves/models/{config.data.dataset}_{step}_{epoch}.pth") 
