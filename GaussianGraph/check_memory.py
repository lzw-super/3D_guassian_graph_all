import torch
import os
import sys
import time

# Add imports from 2D_info_extract.py
from submodules.llava.llava.utils import disable_torch_init
from submodules.llava.llava.model.builder import load_pretrained_model
from submodules.llava.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from submodules.llava.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from submodules.llava.llava.conversation import conv_templates

def print_memory(step):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{step}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    # Also check nvidia-smi for process memory if possible, but python won't easily show it without pynvml
    
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print_memory("Start")

# 1. OpenCLIP
try:
    from dataclasses import dataclass, field
    from typing import Type
    import torchvision
    from torch import nn
    import open_clip
    
    @dataclass
    class OpenCLIPNetworkConfig:
        _target: Type = field(default_factory=lambda: object)
        clip_model_type: str = "ViT-B-16"
        clip_model_pretrained: str = "./submodules/open_clip/open_clip_pytorch_model.bin"
        clip_n_dims: int = 512

    class OpenCLIPNetwork(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            model, _, _ = open_clip.create_model_and_transforms(
                self.config.clip_model_type, 
                self.config.clip_model_pretrained,
                precision="fp16",
            )
            model.eval()
            self.model = model.to("cuda:0")

    print("Loading CLIP...")
    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig())
    print_memory("After CLIP")
except Exception as e:
    print(f"CLIP failed: {e}")

# 2. GroundingDINO
try:
    sys.path.append(os.path.join(os.getcwd(), "submodules", "groundingdino"))
    from submodules.groundingdino.groundingdino.util.inference import Model
    print("Loading GroundingDINO...")
    gsa_config = "./submodules/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gsa_ckpt = "./submodules/groundingdino/groundingdino_swint_ogc.pth"
    grounding_dino_model = Model(model_config_path=gsa_config, model_checkpoint_path=gsa_ckpt, device="cuda:0")
    print_memory("After GroundingDINO")
except Exception as e:
    print(f"GroundingDINO failed: {e}")

# 3. SAM2
try:
    from submodules.segment_anything.sam2.build_sam import build_sam2
    print("Loading SAM2...")
    sam_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_ckpt = "./submodules/segment_anything/checkpoints/sam2.1_hiera_large.pt"
    sam = build_sam2(sam_config, sam_ckpt, "cuda:0", apply_postprocessing=False)
    print_memory("After SAM2")
except Exception as e:
    print(f"SAM2 failed: {e}")
