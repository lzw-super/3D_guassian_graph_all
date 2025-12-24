import torch
import os
import sys

def print_memory(step):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{step}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

print_memory("Start")

print("Importing llava.utils...")
from submodules.llava.llava.utils import disable_torch_init
print_memory("After utils")

print("Importing llava.model.builder...")
from submodules.llava.llava.model.builder import load_pretrained_model
print_memory("After builder")

print("Importing llava.mm_utils...")
from submodules.llava.llava.mm_utils import process_images
print_memory("After mm_utils")
