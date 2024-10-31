import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from memory_tracking import (
    get_nvidia_smi_gpu_memory,
    get_memory_breakdown,
    print_memory_stats,
    plot_memory_usage
)

class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        ## #ORG: All layers were on same device
        ## self.conv_layers = nn.Sequential(...)
        ## self.fc_layers = nn.Sequential(...)
        
        # #Row TP: Split the input width-wise
        # We pad the input from 28x28 to 32x32 because:
        # 1. When split width-wise, each GPU gets 32/2 = 16 columns
        # 2. After first MaxPool2d: 16/2 = 8 columns
        # 3. After second MaxPool2d: 8/2 = 4 columns
        # This ensures clean division at each MaxPool2d operation
        self.pad = nn.ZeroPad2d((2, 2, 2, 2))  # Pad 2 pixels on each side to get 32x32
        
        self.conv_layers_0 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1),    # [B, 1, 32, 16] -> [B, 256, 32, 16]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # -> [B, 512, 32, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> [B, 512, 16, 8]  # Both dimensions halved
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), # -> [B, 1024, 16, 8]
            nn.ReLU(),
            nn.MaxPool2d(2)                                 # -> [B, 1024, 8, 4]  # Both dimensions halved again
        ).to('cuda:0')
        
        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1),    # [B, 1, 32, 16] -> [B, 256, 32, 16]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # -> [B, 512, 32, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> [B, 512, 16, 8]
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), # -> [B, 1024, 16, 8]
            nn.ReLU(),
            nn.MaxPool2d(2)                                 # -> [B, 1024, 8, 4]
        ).to('cuda:1')
        
        self.fc_layers_0 = nn.Sequential(
            nn.Linear(1024 * 8 * 4, 2048),                  # [B, 32768] -> [B, 2048]
            nn.ReLU(),
            nn.Linear(2048, 2048),                          # -> [B, 2048]
            nn.ReLU(),
            nn.Linear(2048, 10)                             # -> [B, 10]
        ).to('cuda:0')
        
        self.fc_layers_1 = nn.Sequential(
            nn.Linear(1024 * 8 * 4, 2048),                  # [B, 32768] -> [B, 2048]
            nn.ReLU(),
            nn.Linear(2048, 2048),                          # -> [B, 2048]
            nn.ReLU(),
            nn.Linear(2048, 10)                             # -> [B, 10]
        ).to('cuda:1')

    def forward(self, x):
        # Pad input from 28x28 to 32x32
        x = self.pad(x)
        
        # Split input width-wise: [B, 1, 32, 32] -> [B, 1, 32, 16] for each GPU
        split_size = x.size(3) // 2  # Split along last dimension (width)
        x0 = x[:, :, :, :split_size].to('cuda:0')  # First half of width
        x1 = x[:, :, :, split_size:].to('cuda:1')  # Second half of width
        
        # Process through parallel conv layers
        x0 = self.conv_layers_0(x0)                # [B, 1024, 8, 4]
        x1 = self.conv_layers_1(x1)                # [B, 1024, 8, 4]
        
        # Flatten and process through FC layers
        x0 = x0.reshape(x0.size(0), -1)           # [B, 1024*8*4]
        x1 = x1.reshape(x1.size(0), -1)           # [B, 1024*8*4]
        
        x0 = self.fc_layers_0(x0)                 # [B, 10]
        x1 = self.fc_layers_1(x1)                 # [B, 10]
        
        # Sum results: [B, 10] + [B, 10] -> [B, 10]
        return x0.to('cuda:1') + x1

def main():
    # Check for multiple GPUs
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires at least 2 GPUs")
    
    # Input data goes to first GPU
    device = torch.device('cuda:0')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create model and move criterion to second GPU
    model = LargeNet()
    criterion = nn.CrossEntropyLoss().to('cuda:1')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining Started")
    print("Memory and Time Statistics")
    print("-" * 50)

    # Memory tracking
    stages = []
    gpu0_mems = []
    gpu1_mems = []
    
    for epoch in range(3):
        epoch_start = time.time()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                # Track memory at different stages
                stages.append('Before Forward')
                gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
                gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))
                
                # Move data to devices
                images = images.to('cuda:0')
                labels = labels.to('cuda:1')
                images = model.pad(images)
                optimizer.zero_grad()
                
                # Split and process through conv layers
                split_size = images.size(3) // 2
                x0 = images[:, :, :, :split_size].to('cuda:0')
                x1 = images[:, :, :, split_size:].to('cuda:1')
                x0 = model.conv_layers_0(x0)
                x1 = model.conv_layers_1(x1)
                
                stages.append('After Conv')
                gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
                gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))
                
                # Process through FC layers
                x0 = model.fc_layers_0(x0.view(x0.size(0), -1))
                x1 = model.fc_layers_1(x1.view(x1.size(0), -1))
                
                stages.append('After FC')
                gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
                gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))
                
                # Combine outputs (sum instead of concatenate)
                outputs = x0.to('cuda:1') + x1
                
                stages.append('After Forward')
                gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
                gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))
                
                loss = criterion(outputs, labels)
                loss.backward()
                stages.append('After Backward')
                gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
                gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))
                
                optimizer.step()
                stages.append('After Optimizer')
                gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
                gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))
                
                # Plot both versions
                plot_memory_usage(stages, gpu0_mems, gpu1_mems, 
                                base_filename='row_tp',
                                include_cuda_overhead=True)
                plot_memory_usage(stages, gpu0_mems, gpu1_mems, 
                                base_filename='row_tp',
                                include_cuda_overhead=False)
            else:
                # Regular training loop
                images = images.to('cuda:0')
                labels = labels.to('cuda:1')
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()

        # Print epoch stats
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}:")
        print(f"Time: {epoch_time:.2f} seconds")
        print(f"Loss: {avg_loss:.4f}")
        
        mem_gpu0 = get_memory_breakdown(model, optimizer, 0)
        mem_gpu1 = get_memory_breakdown(model, optimizer, 1)
        print_memory_stats(mem_gpu0, mem_gpu1)

if __name__ == "__main__":
    main()
