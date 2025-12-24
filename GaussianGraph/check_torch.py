import os
import re
import sys
import json
import random
import argparse
import cv2
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from dataclasses import dataclass, field
from typing import Tuple, Type
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import torchvision
try:
    from torchvision import _meta_registrations
    print("Torchvision _meta_registrations imported successfully")
except Exception as e:
    print(f"Error importing torchvision _meta_registrations: {e}")
