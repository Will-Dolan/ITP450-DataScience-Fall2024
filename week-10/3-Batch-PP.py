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
import torch.cuda.comm as comm

class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        ## #ORG: All layers were on same device
        ## self.conv_layers = nn.Sequential(...)
        ## self.fc_layers = nn.Sequential(...)
        
        # #Naive PP: Split layers across devices
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1),    # [B, 1, 28, 28] -> [B, 256, 28, 28]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # -> [B, 512, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> [B, 512, 14, 14]
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), # -> [B, 1024, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2)                                 # -> [B, 1024, 7, 7]
        ).to('cuda:0')
        
        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 2048),  # [B, 50176] -> [B, 2048]
            nn.ReLU(),
            nn.Linear(2048, 2048),          # -> [B, 2048]
            nn.ReLU(),
            nn.Linear(2048, 10)             # -> [B, 10]
        ).to('cuda:1')

    def forward(self, x):
        ## #ORG: Single device forward pass
        ## x = self.conv_layers(x)
        ## x = x.view(x.size(0), -1)
        ## x = self.fc_layers(x)
        
        # #Naive PP: Multi-device forward pass with transfers
        x = self.conv_layers(x)
        x = x.to('cuda:1')  # Transfer to second GPU
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class PipelineBuffer:
    def __init__(self, num_microbatches):
        self.num_microbatches = num_microbatches
        # Only store activations for active microbatches
        self.conv_outputs = [None] * 2  # Need only 2 slots for F/B overlap
        self.fc_outputs = [None] * 2
        self.losses = [None] * num_microbatches  # Keep all losses for averaging
        self.mb_idx_map = {}  # Maps microbatch idx to buffer slot
        
    def get_slot(self, mb_idx):
        return mb_idx % 2
    
    def clear(self):
        self.conv_outputs = [None] * 2
        self.fc_outputs = [None] * 2
        self.losses = [None] * self.num_microbatches
        self.mb_idx_map.clear()

def process_pipeline_batch(model, images, labels, criterion, optimizer, micro_batch_size, is_first_batch=False):
    num_microbatches = images.size(0) // micro_batch_size
    micro_images = torch.split(images, micro_batch_size)
    micro_labels = torch.split(labels, micro_batch_size)
    
    # Create streams for each GPU
    stream_gpu0 = torch.cuda.Stream(device='cuda:0')
    stream_gpu1 = torch.cuda.Stream(device='cuda:1')
    
    # Create buffer for pipeline stages
    buffer = PipelineBuffer(num_microbatches)
    
    # Memory tracking
    stages = []
    gpu0_mems = []
    gpu1_mems = []
    
    if is_first_batch:
        stages.append('Start')
        gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
        gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))

    # Pipeline schedule
    total_steps = 2 * num_microbatches + 1  # Adjusted steps
    for step in range(total_steps):
        # Calculate indices for this step
        forward_mb_idx = step if step < num_microbatches else None
        backward_mb_idx = step - num_microbatches if step >= num_microbatches else None
        
        # Forward Pass (if applicable)
        if forward_mb_idx is not None:
            with torch.cuda.stream(stream_gpu0):
                imgs = micro_images[forward_mb_idx].to('cuda:0')
                slot = buffer.get_slot(forward_mb_idx)
                buffer.conv_outputs[slot] = model.conv_layers(imgs)
                buffer.mb_idx_map[forward_mb_idx] = slot
            
            with torch.cuda.stream(stream_gpu1):
                stream_gpu1.wait_stream(stream_gpu0)
                slot = buffer.mb_idx_map[forward_mb_idx]
                conv_out = buffer.conv_outputs[slot].to('cuda:1', non_blocking=True)
                conv_out = conv_out.view(conv_out.size(0), -1)
                buffer.fc_outputs[slot] = model.fc_layers(conv_out)
                
                lbls = micro_labels[forward_mb_idx].to('cuda:1')
                buffer.losses[forward_mb_idx] = criterion(buffer.fc_outputs[slot], lbls)

            if is_first_batch and forward_mb_idx == 0:
                stages.append(f'Forward MB{forward_mb_idx}')
                gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
                gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))
        
        # Backward Pass (if applicable)
        if backward_mb_idx is not None and backward_mb_idx < num_microbatches:  # Added boundary check
            with torch.cuda.stream(stream_gpu1):
                slot = buffer.mb_idx_map[backward_mb_idx]
                buffer.losses[backward_mb_idx].backward(retain_graph=True)
                
                # Free up memory
                if backward_mb_idx > 0:  # Keep last activation for optimizer step
                    prev_slot = buffer.mb_idx_map[backward_mb_idx - 1]
                    buffer.conv_outputs[prev_slot] = None
                    buffer.fc_outputs[prev_slot] = None

            if is_first_batch and backward_mb_idx == 0:
                stages.append(f'Backward MB{backward_mb_idx}')
                gpu0_mems.append(get_memory_breakdown(model, optimizer, 0))
                gpu1_mems.append(get_memory_breakdown(model, optimizer, 1))
        
        # Synchronize at the end of each step to maintain ordering
        torch.cuda.synchronize()
    
    # Calculate total loss
    total_loss = sum(buffer.losses) / num_microbatches
    
    return total_loss, stages, gpu0_mems, gpu1_mems

def main():

    # #Naive PP: Check for multiple GPUs
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires at least 2 GPUs")
    
    # #Naive PP: Input data goes to first GPU
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
    criterion = nn.CrossEntropyLoss().to('cuda:1')  # #Naive PP: Loss computation on GPU1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining Started")
    print("Memory and Time Statistics")
    print("-" * 50)

    # Memory tracking
    stages = []
    gpu0_mems = []
    gpu1_mems = []
    
    # Add microbatch size
    micro_batch_size = 16  # Adjust this based on your needs
    
    for epoch in range(3):
        epoch_start = time.time()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            if epoch == 0 and batch_idx == 0:
                loss, stages, gpu0_mems, gpu1_mems = process_pipeline_batch(
                    model, images, labels, criterion, optimizer, 
                    micro_batch_size=16, is_first_batch=True
                )
                
                # Plot memory usage
                plot_memory_usage(stages, gpu0_mems, gpu1_mems, 
                                base_filename='async_pipeline_parallel',
                                include_cuda_overhead=True)
                plot_memory_usage(stages, gpu0_mems, gpu1_mems, 
                                base_filename='async_pipeline_parallel',
                                include_cuda_overhead=False)
            else:
                loss, _, _, _ = process_pipeline_batch(
                    model, images, labels, criterion, optimizer, 
                    micro_batch_size=16, is_first_batch=False
                )
            
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
