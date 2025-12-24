import os
import pickle
import gzip
# Set allocator to avoid fragmentation
# 设置内存分配器配置，避免显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import warnings
# 过滤特定的警告信息
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
import re
import sys
import json
import random
import argparse
import cv2
import copy
import gc
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
# 引入 supervision 库用于可视化
import supervision as sv 

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载被截断的图像文件，防止因图片损坏导致的报错
from dataclasses import dataclass, field
from typing import Tuple, Type
# import open3d as o3d
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import torchvision
from torch import nn
from loguru import logger
try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

# 导入 SAM2 相关模块
# Segment Anything Model 2 (SAM2) 的构建和预测模块
from submodules.segment_anything.sam2.build_sam import build_sam2
from submodules.segment_anything.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from submodules.segment_anything.sam2.sam2_image_predictor import SAM2ImagePredictor

# Add groundingdino to python path to allow internal absolute imports
# 将 GroundingDINO 添加到 Python 路径，以便允许内部绝对导入
# GroundingDINO 是一个开放集目标检测模型，可以通过文本提示检测任意对象
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules", "groundingdino"))

"""
================================================================================
Supervision 库使用说明 (Supervision Library Usage)
================================================================================
本脚本集成 supervision 库用于计算机视觉任务的可视化，核心功能包括：

1. Detections (sv.Detections):
   - 统一的数据结构，用于存储检测结果 (边界框 xyxy, 掩码 mask, 置信度 confidence, 类别 class_id)。
   - 支持从各类模型 (如 YOLO, Transformers) 的输出进行转换。

2. Annotators (标注器):
   - BoxAnnotator: 用于绘制对象边界框。
   - MaskAnnotator: 用于绘制分割掩码 (Instance Segmentation)。
   - LabelAnnotator: 用于在框或掩码旁添加文本标签 (类别名 + 置信度)。

3. 使用示例:
   # 创建 Detections 对象
   detections = sv.Detections(
       xyxy=boxes,           # (N, 4) numpy array
       confidence=scores,    # (N,) numpy array
       class_id=class_ids,   # (N,) numpy array
       mask=masks            # (N, H, W) numpy array (Optional)
   )

   # 初始化标注器
   box_annotator = sv.BoxAnnotator()
   mask_annotator = sv.MaskAnnotator()
   
   # 绘制
   annotated_image = box_annotator.annotate(scene=image, detections=detections)
   annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
================================================================================
"""

from submodules.groundingdino.groundingdino.util.inference import Model
from submodules.llava.llava.utils import disable_torch_init
from submodules.llava.llava.model.builder import load_pretrained_model
from submodules.llava.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from submodules.llava.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from submodules.llava.llava.conversation import conv_templates

"""
================================================================================
使用示例说明 (Usage Example)
================================================================================

1. 输入数据准备要求:
   - 图像文件夹: 包含待处理的 RGB 图像 (支持 .jpg, .png 等格式)。
   - 目录结构建议:
     /path/to/dataset/
       ├── color/ (或 images/) -> 存放原始图像
       └── language_features/  -> 脚本生成的输出将保存在这里

2. 典型参数配置建议:
   - --dataset_path: 数据集根目录。
   - --config: SAM2 模型配置文件路径。
   - --sam_ckpt: SAM2 预训练权重路径。
   - --gsa_config: GroundingDINO 配置文件路径。
   - --gsa_ckpt: GroundingDINO 权重路径。
   - --llava_ckpt: LLaVa 模型权重路径。
   - --box_threshold: 检测框置信度阈值 (建议 0.2-0.4)。
   - --text_threshold: 文本匹配阈值 (建议 0.2-0.4)。

3. 预期输出格式说明:
   - 在 output_dir 或 language_features 目录下生成:
     - *_s.npy: 分割掩码 (Segmentation Maps)。
     - *_f.npy: CLIP 特征向量 (Feature Embeddings)。
     - *_r.json: 场景图数据 (Scene Graph)，包含对象类别、描述、关系等。

4. 常见问题处理建议:
   - 显存不足: 尝试减小 batch size (如 SAM 生成器的 points_per_batch) 或启用 --skip_relations。
   - 检测不到对象: 降低 --box_threshold 和 --text_threshold。
   - LLaVa 回复为空: 检查网络连接或模型路径是否正确。

================================================================================
""" 
import json 
def extract_unique_categories(json_file_path):
    """
    从给定的JSON文件中提取所有唯一的物体类别
    
    参数:
    json_file_path: JSON文件的路径
    
    返回:
    set: 包含所有唯一物体类别的集合
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 提取所有节点的类别
    categories = set()
    for node in data.get('nodes', []):
        category = node.get('category')
        if category:
            categories.add(category)
    
    return categories
categories = extract_unique_categories('/home/zhengwu/Desktop/3D-Gaussian/office_scene_graph_70000.json') 
@dataclass
class OpenCLIPNetworkConfig:
    """
    OpenCLIP 网络配置类
    定义了 CLIP 模型的类型、预训练权重路径和嵌入维度
    """
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"  # CLIP 模型架构类型
    clip_model_pretrained: str = "./submodules/open_clip/open_clip_pytorch_model.bin"  # 预训练权重文件路径
    clip_n_dims: int = 512  # CLIP 特征向量的维度
   
class OpenCLIPNetwork(nn.Module):
    """
    OpenCLIP 网络封装类，用于提取图像和文本特征。
    主要功能是将图像块 (Image Patches) 或文本转换为高维特征向量。
    """
    def __init__(self, config: OpenCLIPNetworkConfig):
        """
        初始化 OpenCLIP 模型
        """
        super().__init__()
        self.config = config
        # 图像预处理流程：
        # 1. Resize: 将图像调整为 224x224 (CLIP 的标准输入尺寸)
        # 2. Normalize: 使用 CLIP 预训练时的均值和标准差进行归一化
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        
        # 创建 OpenCLIP 模型实例
        # precision="fp16" 使用半精度浮点数以节省显存和加速计算
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type, 
            self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()  # 设置为评估模式，冻结 BatchNorm 和 Dropout
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)  # 获取对应的文本分词器
        self.model = model.to(args.device)  # 将模型移动到指定计算设备 (如 cuda:0)
        self.clip_n_dims = self.config.clip_n_dims

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def encode_image(self, input):
        """
        编码图像，返回图像特征
        输入: (Batch_Size, 3, H, W) 的图像张量
        输出: (Batch_Size, Embedding_Dim) 的特征向量
        """
        processed_input = self.process(input).half()  # 应用预处理并将数据转换为 fp16
        return self.model.encode_image(processed_input)
    
    def encode_texts(self, class_ids, classes):
        """
        编码文本，返回文本特征
        输入: 类别 ID 列表和类别名称列表
        输出: 归一化后的文本特征向量
        """
        with torch.no_grad():
            # 将文本转换为 Token ID
            tokenized_texts = torch.cat([self.tokenizer(classes[class_id]) for class_id in class_ids]).to(args.device)
            # 提取文本特征
            text_feats = self.model.encode_text(tokenized_texts)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)  # 对特征向量进行 L2 归一化
        return text_feats

class LLaVaChat():
    """
    LLaVa (Large Language-and-Vision Assistant) 聊天模型封装类。
    用于处理多模态查询，结合图像内容和文本提示生成自然语言描述。
    """
    # Model Constants - 定义特殊 Token
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER = "<image-placeholder>"

    def __init__(self, model_path):
        """
        初始化 LLaVaChat 模型
        :param model_path: 模型权重的本地路径
        """
        disable_torch_init() # 禁用某些 Torch 初始化以加速加载

        self.model_name = get_model_name_from_path(model_path)  
        # 加载预训练模型、分词器和图像处理器
        # load_4bit=True: 使用 4-bit 量化加载模型，显著降低显存占用
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, self.model_name, device="cuda", load_4bit=True)

        # 根据模型名称自动选择对应的对话模板 (Conversation Template)
        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

    def preprocess_image(self, images):
        """
        预处理图像以适配 LLaVa 模型输入
        """
        x = process_images(
            images,
            self.image_processor,
            self.model.config)

        return x.to(self.model.device, dtype=torch.float16)

    def __call__(self, query, image_features, image_sizes):
        """
        前向推理函数：调用模型生成回答
        :param query: 文本提示 (Prompt)
        :param image_features: 预处理后的图像特征
        :param image_sizes: 原始图像尺寸列表
        :return: 模型生成的文本回答
        """
        # Given this query, and the image_featurese, prompt LLaVA with the query,
        # using the image_features as context.

        conv = conv_templates[self.conv_mode].copy()

        # 构建包含特殊 Token 的输入提示
        if self.model.config.mm_use_im_start_end:
            inp = LLaVaChat.DEFAULT_IM_START_TOKEN +\
                  LLaVaChat.DEFAULT_IMAGE_TOKEN +\
                  LLaVaChat.DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            inp = LLaVaChat.DEFAULT_IMAGE_TOKEN + '\n' + query
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 将提示文本转换为 token IDs
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, LLaVaChat.IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.temperature = 0 # 温度设为 0 以获得确定性输出
        self.max_new_tokens = 512 # 最大生成长度
        with torch.inference_mode():
            # 生成回答
            output_ids = self.model.generate(
                input_ids,
                images=image_features,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        return outputs

def describe_LLAVA(mask_id, image, chat:LLaVaChat, class_i, class_j, Cord_i, Cord_j, mode):
    """
    核心功能函数：使用 LLaVa 模型生成图像描述或分析对象关系
    根据不同的 'mode' 构建不同的 Prompt 与模型交互。
    
    :param mask_id: 掩码 ID (或 ID 对)
    :param image: 输入图像 (可以是裁剪后的对象图或全图)
    :param chat: LLaVaChat 实例
    :param class_i: 对象 i 的类别名称
    :param class_j: 对象 j 的类别名称 (仅关系模式需要)
    :param Cord_i: 对象 i 的边界框 [x1, y1, x2, y2]
    :param Cord_j: 对象 j 的边界框 (仅关系模式需要)
    :param mode: 模式 ("category", "captions", "relationships")
    :return: 包含描述信息的字典
    """

    ### caption
    image_sizes = [image.size]
    image_tensor = chat.preprocess_image([image]).to("cuda", dtype=torch.float16)
    template = {}

    if mode == "category":
        # 模式1: 类别发现
        # 询问模型图像中有哪些主要对象类别，用于后续 GroundingDINO 的检测
        query_base = """Identify and list only the main object categories clearly visible in the image."""

        query_tail = """
        Provide only the category names, separated by commas.
        Only list the main object categories in the image.
        Maximum 10 categories, focus on clear, foreground objects
        Each category should be listed only once, even if multiple instances of the same category are present.
        Avoid overly specific or recursive descriptions.
        Do not include descriptions, explanations, or duplicates.
        Do not include quotes, brackets, or any additional formatting in the output.
        Examples:
        Chair, Table, Window
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
        # 清理输出文本
        template["categories"] = re.sub(r'\s+', ' ', text.replace("<s>", "").replace("</s>", "").replace("-", "").strip())

    elif mode == "captions":
        # 模式2: 详细描述
        # 对单个对象进行外观、材质等详细描述
        query_base = """Describe the visible object in front of you, 
        focusing on its spatial dimensions, visual attributes, and material properties."""
        
        query_tail = """
        The object is typically found in indoor scenes and its category is {class_i}.
        Briefly describe the object within ten word. Keep the description concise.
        Focus on the object's appearance, geometry, and material. Do not describe the background or unrelated details.
        Ensure the description is specific and avoids vague terms.
        Examples: 
        a closed wooden door with a glass panel;
        a pillow with a floral pattern;
        a wooden table;
        a gray wall.
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query.format(class_i=class_i), image_features=image_tensor, image_sizes=image_sizes)
        template["id"] = mask_id
        template["description"] = text.replace("<s>", "").replace("</s>", "").strip()

    elif mode == "relationships":
        # 模式3: 关系分析
        # 分析两个对象之间的空间位置关系 (如 Above, Left, Next to)
        query_base = """There are two objects with category and 2D coordinate, 
        paying close attention to the positional relationship between two selected objects."""
        query_tail = """
        You are capable of analyzing spatial relationships between objects in an image.

        In the given image, there are two boxed objects:
        - The object selected by the red box is [{class_i}], and its bounding box coordinates are {bbox1}.
        - The object selected by the blue box is [{class_j}], and its bounding box coordinates are {bbox2}.

        Note: The bounding box coordinates are in the format (x_min, y_min, x_max, y_max), where (x_min, y_min) represents the top-left corner of the box and (x_max, y_max) represents the bottom-right corner of the box.

        The spatial relationship between [{class_i}] and [{class_j}] may include, but is not limited to, the following types:
        - "Above" means Object A is located higher in vertical position (y_min smaller).
        - "Below" means Object A is located lower in vertical position (y_min larger).
        - "Left" means Object A's x_min is smaller than Object B's x_min.
        - "Right" means Object A's x_min is larger than Object B's x_min.
        - "Inside" means Object A's bounding box is fully contained within Object B's bounding box.
        - "Contains" means Object A's bounding box fully contains Object B's bounding box.
        - "Next to" means the distance between boxes is very small, without overlap.

        Please provide the output in the following format:
        Coarse: The spatial relationship between {class_i} and {class_j}; Fine: A detailed description of the relationship (optional).

        Example output:
        Coarse: The cup is on the table; Fine: The cup is resting near the center of the table, with its handle facing outward.
        Coarse: The book is under the lamp; Fine: The book lies directly beneath the lamp, slightly tilted, as if recently placed.
        Coarse: The cat is next to the sofa; Fine: The cat is sitting closely beside the sofa's left armrest, partially leaning on it.
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query.format(class_i=class_i, class_j=class_j, bbox1=Cord_i, bbox2=Cord_j), image_features=image_tensor, image_sizes=image_sizes)
        template["id_pair"] = mask_id
        template["relationship"] = text.replace("<s>", "").replace("</s>", "").strip()

    return template

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    执行掩码非极大值抑制 (Mask NMS)，去除重叠过高或置信度过低的掩码。
    
    Args:
        masks (torch.Tensor): 掩码张量，形状 (num_masks, H, W)
        scores (torch.Tensor): 掩码得分，形状 (num_masks,)
        iou_thr (float, optional): IoU 阈值，超过此值的重叠掩码将被抑制。
        score_thr (float, optional): 分数阈值，低于此分数的掩码将被丢弃。
        inner_thr (float, optional): 内部包含阈值，用于处理一个掩码包含另一个掩码的情况。
        **kwargs: 其他参数。
    Returns:
        selected_idx (torch.Tensor): 保留的掩码索引。
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            # 处理包含关系：如果一个掩码大部分被另一个掩码包含
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # 保证至少保留前3个掩码，防止所有掩码都被过滤掉
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    # 基于分数和掩码之间的重叠率去除冗余掩码
    masks_new = ()
    for masks_lvl in (args):
        if not masks_lvl:
            masks_new += ([],)  # 或者其他适当的处理
            continue
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def get_seg_img(mask, image):
    """
    根据掩码提取分割区域图像，背景置黑
    """
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    """
    将图像填充为正方形，保持长宽比，用于 CLIP 输入预处理
    """
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def compute_iou_matrix(generated_masks, seg_map, unique_mask_indices):
    """
    计算所有生成的掩码与seg_map中所有掩码的IoU
    :param generated_masks: (num_masks, H, W)，生成的掩码数组
    :param seg_map: (H, W), 全图的seg_map
    :param unique_mask_indices: seg_map中唯一的掩码索引
    :return: (num_masks, num_seg_masks) 的IoU矩阵
    """
    num_seg_masks = len(unique_mask_indices)
    generated_masks = generated_masks.astype(np.bool_)
    # 初始化一个空的IoU矩阵
    iou_matrix = np.zeros((1, num_seg_masks))
    
    # 逐个计算IoU
    for i, mask_index in enumerate(unique_mask_indices):
        if mask_index == -1:  # 跳过背景
            continue
        
        # 获取seg_map中当前掩码的区域
        seg_mask = (seg_map == mask_index)  # (H, W)
        
        # 计算交集和并集
        intersection = np.sum(generated_masks & seg_mask)  # (num_masks, H, W) 与 (H, W) 计算交集
        union = np.sum(generated_masks | seg_mask)  # (num_masks, H, W) 与 (H, W) 计算并集
        
        # 计算IoU
        iou_matrix[:, i] = intersection / (union + 1e-6)  # 防止除以零，1e-6为小常数

    return iou_matrix

def get_bbox_img(box, image):
    """
    根据边界框提取图像区域
    """
    image = image.copy()
    x_min, y_min, x_max, y_max = map(int, box)
    # 从图像中截取框内区域
    seg_img = image[y_min:y_max, x_min:x_max]
    return seg_img

def sam_predictor(seg_map, image, detections):  
    """
    SAM 模型预测函数：使用 SAM 模型根据检测框 (Prompt) 预测精细分割掩码
    
    流程：
    1. 接收 GroundingDINO 检测到的边界框 (detections)。
    2. 设置 SAM 预测器的图像编码 (predictor_sam.set_image)。
    3. 对每个检测框，提示 SAM 生成对应的分割掩码。
    4. 将生成的掩码与全局分割图 (seg_map) 进行匹配，找到对应的 ID。
    
    :param seg_map: 全局分割图字典 (不同尺度)
    :param image: 输入图像 (numpy array)
    :param detections: 检测结果对象 (GroundingDINO output)，包含 xyxy, class_id, confidence
    :return: 
        - seg_imgs: 裁剪并填充后的对象图像张量 (Batch, 3, 224, 224)
        - categories: 类别 ID 列表
        - mask_indices: 匹配到的全局掩码索引字典 
        - all_masks: 生成的所有分割掩码 (N, H, W)
    """
    # 过滤掉 class_id 为 None 的检测  
    valid_mask = np.array([cid is not None for cid in detections.class_id])  
      
    if not valid_mask.any():  
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long), {}, np.empty((0, image.shape[0], image.shape[1]), dtype=bool)
      
    # 只保留有效的检测  
    detections.xyxy = detections.xyxy[valid_mask]  
    detections.class_id = detections.class_id[valid_mask]  
    detections.confidence = detections.confidence[valid_mask]  
      
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # SAM 图像编码：这是一次性操作，提取图像特征
        predictor_sam.set_image(image=image)  
        seg_img_list = []  
        classes_list = []  
        mask_indices = {}   
        mask_list = [] # 用于收集生成的掩码
  
        unique_mask_indices_cache = {}  
        for mode in seg_map.keys():  
            unique_mask_indices_cache[mode] = np.unique(seg_map[mode])  
  
        # 使用 enumerate 确保索引从0开始连续  
        for i, box in enumerate(detections.xyxy):  
            category_id = detections.class_id[i]  
            classes_list.append(category_id)  
            
            # 预测掩码: 将检测框作为 Box Prompt 输入给 SAM
            # multimask_output=True: 允许 SAM 输出多个候选掩码，通常取分数最高的一个
            masks, scores, logits = predictor_sam.predict(box=box, multimask_output=True)  
            index = np.argmax(scores) # 选择置信度最高的掩码
            generated_mask = masks[index]  
            mask_list.append(generated_mask) # 收集掩码
              
            mode_mask_indices = {}  
            for mode, unique_mask_indices in unique_mask_indices_cache.items():  
                # 计算生成的掩码与现有全局分割图的 IoU，找到最佳匹配的 segment ID
                iou_matrix = compute_iou_matrix(generated_mask[None, :, :], seg_map[mode], unique_mask_indices)  
                best_mask_index = unique_mask_indices[np.argmax(iou_matrix)]  
                mode_mask_indices[mode] = best_mask_index  
  
            mask_indices[i] = mode_mask_indices  # 使用 i 作为键  
  
            # 提取对象图像块，用于后续 CLIP 特征提取
            seg_img = get_bbox_img(box, image)  
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))  
            seg_img_list.append(pad_seg_img)  
        all_masks = np.array(mask_list) if len(mask_list) > 0 else np.empty((0, image.shape[0], image.shape[1]), dtype=bool)
        if len(classes_list) > 0:  
            categories = torch.tensor(classes_list, dtype=torch.long)  
            seg_imgs = np.stack(seg_img_list, axis=0)  
            # 归一化并转换为 Tensor (B, C, H, W)
            seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to(args.device)  
  
    return seg_imgs, categories, mask_indices ,all_masks

def sam_encoder(image, sam_model, args):
    """
    SAM 全局分割函数：使用 SAM 自动掩码生成器生成多尺度分割掩码
    这不依赖于检测框，而是自动在图像上撒点进行网格化分割。
    
    :param image: 输入图像
    :param sam_model: SAM2 模型实例
    :param args: 参数对象
    :return: 
        - seg_images: 包含不同尺度下每个分割区域的裁剪图像字典
        - seg_maps: 包含不同尺度下全图分割索引图字典
    """
    
    # Initialize generators for different scales locally to avoid memory leak
    # 在函数内部初始化生成器，以避免显存泄漏
    points_s = 32 if args.fast_sam else 64
    
    # 使用较小的 points_per_batch 以节省显存
    points_per_batch = 16
    
    mask_generator_s = SAM2AutomaticMaskGenerator(model=sam_model, points_per_side=points_s, points_per_batch=points_per_batch)
    mask_generator_m = SAM2AutomaticMaskGenerator(model=sam_model, points_per_side=32, points_per_batch=points_per_batch)
    mask_generator_l = SAM2AutomaticMaskGenerator(model=sam_model, points_per_side=16, points_per_batch=points_per_batch)

    # pre-compute masks
    # masks_default = mask_generator.generate(image)
    # 生成不同尺度的掩码 (Small, Medium, Large)
    masks_s = mask_generator_s.generate(image)
    masks_m = mask_generator_m.generate(image)
    masks_l = mask_generator_l.generate(image)
    masks_default = masks_m # Reuse m for default

    # Cleanup generators
    del mask_generator_s, mask_generator_m, mask_generator_l
    torch.cuda.empty_cache()

    # pre-compute postprocess
    #masks_default, masks_s, masks_m, masks_l = masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.7, score_thr=0.6, inner_thr=0.5)
   
    def mask2segmap(masks, image):
        """
        辅助函数：将 SAM 输出的掩码列表转换为分割索引图 (Label Map)
        """
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            # 提取分割区域图像
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            # 填充分割图
            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
      
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map
    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def is_overlapping(box1, box2):
    """
    Check if two bounding boxes overlap.
    检查两个边界框是否重叠。
    
    Args:
        box1 (list or array): Coordinates of the first box [x1, y1, x2, y2].
        box2 (list or array): Coordinates of the second box [x1, y1, x2, y2].
    
    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check if there is no overlap
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    return True

def object_pairs(box1, box2):
    """
    检查两个对象是否重叠，并计算中心距离
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    center1 = torch.tensor([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    center2 = torch.tensor([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
    dist = torch.norm(center1 - center2, p=2)

    # Check if there is no overlap
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        overlapping = False
    else:
        overlapping = True
    
    return overlapping, dist

def crop_and_blackout(image, bbox1, bbox2, padding):
    """
    从图像中截取指定索引的两个矩形框区域，并将其余部分设为黑色。
    
    参数：
        image: 输入图像 (H, W, C)
        detections: 检测框列表，每个元素是一个 [x1, y1, x2, y2]
        idx1: 第一个矩形框的索引
        idx2: 第二个矩形框的索引
        
    返回：
        cropped_image: 包含两个矩形框的图像，其他区域为黑色
    """
    height, width = image.shape[:2]
    # 复制图像，初始化为黑色图像
    cropped_image = np.zeros_like(image)
    
    # 获取第一个矩形框的坐标
    x1, y1, x2, y2 = map(int, bbox1)
    # 扩充裁剪区域，确保不会超出图像边界
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    # 将第一个矩形框区域复制到黑色图像中
    cropped_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    # 获取第二个矩形框的坐标
    x1, y1, x2, y2 = map(int, bbox2)
    # 扩充裁剪区域，确保不会超出图像边界
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    # 将第二个矩形框区域复制到黑色图像中
    cropped_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    return cropped_image

def draw_bounding_boxes(image, bbox1, bbox2, color1=(0, 255, 0), color2=(0, 0, 255), thickness=2):
    """
    在图像上绘制两个矩形框。
    
    参数：
        image: 输入图像 (H, W, C)
        bbox1: 第一个矩形框的坐标 [x1, y1, x2, y2]
        bbox2: 第二个矩形框的坐标 [x1, y1, x2, y2]
        color1: 第一个矩形框的颜色 (B, G, R)
        color2: 第二个矩形框的颜色 (B, G, R)
        thickness: 矩形框的线条粗细
    """
    x1_min, y1_min, x1_max, y1_max = map(int, bbox1)
    x2_min, y2_min, x2_max, y2_max = map(int, bbox2)
    
    # 绘制第一个矩形框
    cv2.rectangle(image, (x1_min, y1_min), (x1_max, y1_max), color1, thickness)
    
    # 绘制第二个矩形框
    cv2.rectangle(image, (x2_min, y2_min), (x2_max, y2_max), color2, thickness)

def graph_construct(image_path, sam_predictor, sam_model, llava_chat, classes_set, global_class_counts,global_class, args):
    """
    构建图像的场景图，包括对象检测、分割、描述和关系提取
    核心流程：
    1. SAM Encoder: 生成全局多尺度分割图。
    2. LLaVa: 识别图像中的主要类别。
    3. GroundingDINO: 根据类别文本提示检测对象边界框。
    4. SAM Predictor: 根据边界框生成精细掩码，并与全局分割图匹配。
    5. LLaVa: 为检测到的对象生成详细描述。
    6. LLaVa: 分析对象之间的空间关系。

    :param image_path: 图像路径
    :param sam_predictor: SAM 预测器 (Box Prompt -> Mask)
    :param sam_model: SAM 模型实例 (用于 Auto Mask Generation)
    :param llava_chat: LLaVa 聊天模型
    :param classes_set: 类别集合 (用于收集所有出现的类别)
    :param args: 参数对象
    :return: CLIP 嵌入, 分割图, 场景图字典
    """
    
    image_pil = Image.open(image_path).convert("RGB")
    image = cv2.imread(image_path)
    # global resolution
    # resolution = (800, 800)   
    # resolution = (image.shape[0], image.shape[1])   
    if image.shape != resolution : 
        image = cv2.resize(image, resolution)
    # image_pil = image_pil.resize((resolution[1], resolution[0]), Image.LANCZOS) 
    image_pil = image_pil.resize((resolution[0], resolution[1]), Image.LANCZOS)

    # 1. 使用 SAM 编码器生成多尺度分割图
    # 这一步生成的是无语义的分割块
    seg_images, seg_map = sam_encoder(np.array(image_pil), sam_model, args)

    clip_embeds = {}
    for mode in seg_images.keys():  # 动态获取实际存在的层级
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            # 使用 CLIP 编码图像块
            clip_embed = clip_model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()

    graph_dict = {}
    print(image_path, '******************') 
    if not args.use_known_classes:
        with torch.no_grad():
            # 2. 使用 LLaVa 识别图像中的主要对象类别
            # 输出示例: ['chair', 'table', 'monitor']
            classes_info = describe_LLAVA(mask_id=None, image=image_pil, chat=llava_chat, 
                                        class_i=None, class_j=None, Cord_i=None, Cord_j=None, mode='category')
            classes = list(set(classes_info['categories'].strip('"').split(',')))
            classes = [item.strip().replace(',', '') for item in classes]
            print(classes, 'class')
            classes_set.update(classes)

            # grounding_dino detector
            if len(classes) > 0:
                classes = classes
            else:
                assert len(classes) == 0, "Error: No target detected in the image!"

            graph_dict['classes'] = classes
    else :  
        background_objects = {'Wall', 'Ceiling', 'Floor'}
        classes_set = extract_unique_categories('/home/zhengwu/Desktop/3D-Gaussian/office_scene_graph_70000.json')  
        foreground_objects = classes_set - background_objects  
        classes = list(classes_set)
        graph_dict['classes'] = list(classes_set)
        graph_dict['foreground_objects'] = list(foreground_objects) 
        graph_dict['background_objects'] = list(background_objects) 


    # 3. 使用 GroundingDINO 检测对象
    # GroundingDINO 模型调用部分：
    # - image: 输入图像 (BGR 格式)
    # - classes: 文本提示词列表 (Text Prompts)，模型将寻找这些类别的对象
    # - box_threshold: 框置信度阈值，过滤低置信度框
    # - text_threshold: 文本匹配阈值  
    # global global_class
    global_class.update(classes)
    detections = grounding_dino_model.predict_with_classes(
        image=image, # This function expects a BGR image...
        classes=graph_dict['classes'],
        box_threshold=0.4,
        text_threshold=0.4,
    )

    # 在第 593 行之后添加  
    print(f"Detections data: {detections.data}")  
    print(f"Detections metadata: {detections.metadata}")  
    # 检查是否有 class_name 或其他类别信息  
    if hasattr(detections, 'data') and detections.data is not None:  
        for key, value in detections.data.items():  
            print(f"  data['{key}']: {value}")
    
    if len(detections.class_id) > 0:
        ### Non-maximum suppression ###
        # 执行非极大值抑制 (NMS)
        # 消除重叠的检测框，只保留置信度最高的框
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            args.nms_threshold
        ).numpy().tolist()
        # print(f"After NMS: {len(detections.xyxy)} boxes")

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        
        # Somehow some detections will have class_id=-1, remove them
        # 移除无效的 class_id
        valid_idx = detections.class_id != -1
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]

    else:
        # 如果没有检测到对象，尝试降低阈值重新检测
        # 这是一个容错机制，防止因阈值过高导致漏检
        detections = grounding_dino_model.predict_with_classes(
        image=image, # This function expects a BGR image...
        classes=classes,
        box_threshold=0.2,
        text_threshold=0.2,
        )

        if len(detections.class_id) == 0:
            assert len(detections.class_id) == 0, "Error: No target detected in the image!"

        valid_idx = detections.class_id != -1
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]

    # GroundingDINO 
    print(f"GroundingDINO 检测结果:")  
    print(f"  边界框数量: {len(detections.xyxy)}")  
    print(f"  class_id: {detections.class_id}")  
    print(f"  confidence: {detections.confidence}")

    # sam segmentation
    # 4. 模型协同工作机制：使用 SAM 预测器细化分割掩码
    # - GroundingDINO 提供了粗略的边界框 (Box Prompts)
    # - SAM 利用这些边界框生成精确的像素级掩码
    # - seg_map 是全局的分割图，用于将生成的掩码映射到全局 ID
    seg_bbox, categories, match_indices, all_masks = sam_predictor(seg_map, image, detections)

    # 帮我给每张图新增一个results 变量
    if len(detections.xyxy) > 0:
        # 计算特征
        # Compute features using CLIP
        with torch.no_grad():
            image_feats = clip_model.encode_image(seg_bbox)
            image_feats /= image_feats.norm(dim=-1, keepdim=True)
            text_feats = clip_model.encode_texts(detections.class_id, classes)

        # 统计全局出现的所有类别物体
        # Update global class statistics
        for cid in detections.class_id:
            if 0 <= cid < len(classes):
                cname = classes[cid]
                global_class_counts[cname] = global_class_counts.get(cname, 0) + 1

        results = { 
            "xyxy": detections.xyxy, 
            "confidence": detections.confidence, 
            "class_id": detections.class_id, 
            "mask": all_masks, 
            "classes": classes, 
            "image_crops": seg_bbox.cpu().numpy(), 
            "image_feats": image_feats.cpu().numpy(), 
            "text_feats": text_feats.cpu().numpy(), 
        } 
        
        save_dataset_path = os.path.join("/home/zhengwu/Desktop/3D-Gaussian/GaussianGraph/dataset", 'gsa_results')
        if not os.path.exists(save_dataset_path):
            os.makedirs(save_dataset_path)
        
        filename = os.path.basename(image_path)
        file_id = os.path.splitext(filename)[0]
        detections_save_path = os.path.join(save_dataset_path, f"{file_id}.pkl.gz")

        # 使用pickle保存检测结果，并用gzip压缩 
        # Save detection results with pickle and gzip
        try:
            with gzip.open(detections_save_path, "wb") as f: 
                pickle.dump(results, f)
            print(f"Saved results to {detections_save_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

    # ==================================================================================================
    # Visualization Logic using Supervision
    # ==================================================================================================
    # import os 
    try:
        # 准备输出目录 
        out_path = f"/home/zhengwu/Desktop/3D-Gaussian/GaussianGraph/output/2D_output_{resolution[0]}_{resolution[1]}" 
        if not os.path.exists(out_path) : 
            os.makedirs(out_path)
        detection_out_dir = os.path.join(out_path,"detection_out")
        mask_out_dir = os.path.join(out_path,"mask_out")
        os.makedirs(detection_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        # 获取文件名
        filename = os.path.basename(image_path)
        
        # 构建 Supervision Detections 对象
        # 注意: detections.class_id 是整数索引，需要映射回类别名称
        if len(detections.xyxy) > 0:
            # 转换类别 ID 为名称标签
            # detections.class_id 应该与 all_masks 长度一致
            # sam_predictor 内部可能过滤了 detections，所以这里使用的 detections 应该是 sam_predictor 修改后的
            
            # 确保 all_masks 和 detections 对齐
            if len(all_masks) == len(detections.xyxy):
                all_masks_bool = all_masks.astype(bool)
                sv_detections = sv.Detections(
                    xyxy=detections.xyxy,
                    confidence=detections.confidence,
                    class_id=detections.class_id,
                    mask=all_masks_bool
                )

                # 生成标签文本: "Category 0.95"
                labels = [
                    f"{classes[class_id]} {confidence:.2f}"
                    for class_id, confidence
                    in zip(detections.class_id, detections.confidence)
                ]

                # 1. 可视化物体检测框 (Bounding Boxes)
                box_annotator = sv.BoxAnnotator(thickness=2) # 2   8
                label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)  # 0.5   2
                
                annotated_image = image.copy() # image is BGR
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=sv_detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=sv_detections, labels=labels)
                
                cv2.imwrite(os.path.join(detection_out_dir, filename), annotated_image)
                print(f"Saved detection visualization to {os.path.join(detection_out_dir, filename)}")

                # 2. 可视化掩码 (Masks)
                mask_annotator = sv.MaskAnnotator(opacity=0.5)
                
                annotated_image_mask = image.copy()
                annotated_image_mask = mask_annotator.annotate(scene=annotated_image_mask, detections=sv_detections)
                # 叠加框和标签以便更清晰
                annotated_image_mask = box_annotator.annotate(scene=annotated_image_mask, detections=sv_detections)
                annotated_image_mask = label_annotator.annotate(scene=annotated_image_mask, detections=sv_detections, labels=labels)
                
                cv2.imwrite(os.path.join(mask_out_dir, filename), annotated_image_mask)
                print(f"Saved mask visualization to {os.path.join(mask_out_dir, filename)}")
            else:
                print(f"Warning: Detections ({len(detections.xyxy)}) and Masks ({len(all_masks)}) count mismatch. Skipping visualization.")
        else:
                print(f"No detections to visualize for {filename}")

    except Exception as e:
        print(f"Error during visualization for {image_path}: {e}")
        import traceback
        traceback.print_exc()
    # ==================================================================================================

    # 添加这个检查  
    if len(match_indices) == 0:  
        print("Warning: No valid detections after filtering. Skipping captions and relations.")  
        graph_dict['captions'] = []  
        graph_dict['relations'] = []  
        return clip_embeds, seg_map, graph_dict  
    # clip
    tiles = seg_bbox.to(args.device)
    categories = categories.to(args.device)

    # captions of foreground objects
    # 5. 生成前景对象的描述
    descriptions = []
    for idx, fore_box in enumerate(detections.xyxy):
        cropped_image = np.zeros_like(image_pil)

        # 获取矩形框的坐标
        x1, y1, x2, y2 = map(int, fore_box)
        cropped_image[y1:y2, x1:x2] = np.array(image_pil)[y1:y2, x1:x2]
        cropped_image = Image.fromarray(cropped_image)
        
        match_idx = {}
        for mode in match_indices[idx].keys():
            match_idx[mode] = int(match_indices[idx][mode])

        class_i = classes[detections.class_id[idx]]
        
        if not args.skip_captions:
            description = describe_LLAVA(mask_id=match_idx, image=cropped_image, chat=llava_chat, 
                                            class_i=class_i, class_j=None, Cord_i=None, Cord_j=None, mode='captions')
            description['description'] = description.get('description', '')
        else:
            description = {"id": match_idx, "description": f"A {class_i}"}
        
        # 将description中的id改成class_id   $$$
        description['class_id'] = int(detections.class_id[idx])
        description["class_name"] = class_i

        descriptions.append(description)

    graph_dict['captions'] = descriptions
 
    # image_embed = clip_model.encode_image(tiles)
    # image_embed /= image_embed.norm(dim=-1, keepdim=True) 

    # text_embed = clip_model.encode_texts(categories, classes)

    # generate relation
    # 6. 生成对象间的关系
    relations = []
    if not args.skip_relations:
        candidate_pairs = []
        image_height, image_width = image.shape[:2]
        image_diag = torch.sqrt(torch.tensor(image_width ** 2 + image_height ** 2))
        
        # 1. Collect all valid pairs
        # 1. 收集所有有效的一对对象
        for idx_i, bbox_i in enumerate(detections.xyxy):
            for idx_j, bbox_j in enumerate(detections.xyxy[idx_i + 1:], start=idx_i + 1):
                inter, dist = object_pairs(detections.xyxy[idx_i], detections.xyxy[idx_j])
                if inter or dist < args.relation_threshold * image_diag:
                    # Priority: Overlapping first (dist=0 effectively), then by distance
                    priority = 0 if inter else dist.item()
                    candidate_pairs.append({
                        'idx_i': idx_i, 'idx_j': idx_j, 
                        'dist': priority, 'inter': inter
                    })

        # 2. Sort by distance/priority
        candidate_pairs.sort(key=lambda x: x['dist'])

        # 3. Limit number of relations
        if args.max_relations > 0 and len(candidate_pairs) > args.max_relations:
            print(f"Limiting relations from {len(candidate_pairs)} to {args.max_relations}")
            candidate_pairs = candidate_pairs[:args.max_relations]

        # 4. Process selected pairs
        # 4. 处理选定的对
        for pair in candidate_pairs:
            idx_i, idx_j = pair['idx_i'], pair['idx_j']
            torch.cuda.empty_cache()
            
            class_i, class_j = classes[detections.class_id[idx_i]], classes[detections.class_id[idx_j]]

            match_idx_i = {}
            match_idx_j = {}
            for mode in match_indices[idx_i].keys():  
                match_idx_i[mode] = int(match_indices[idx_i][mode])  
            for mode in match_indices[idx_j].keys():  
                match_idx_j[mode] = int(match_indices[idx_j][mode])
            
            image_copy = image.copy()
            draw_bounding_boxes(image_copy, detections.xyxy[idx_i], detections.xyxy[idx_j])
            boxed_image = Image.fromarray(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
            
            output_path = os.path.join(args.output_dir, f"object_i{idx_i}_j{idx_j}.png")
            os.makedirs(args.output_dir, exist_ok=True)
            # boxed_image.save(output_path) # Optional: comment out to save disk IO time

            relation_info = describe_LLAVA((match_idx_i, match_idx_j), boxed_image, llava_chat, 
                                            class_i, class_j, detections.xyxy[idx_i], detections.xyxy[idx_j], mode='relationships')
            print(relation_info)

            # 修改relation_info中的id_pair为class_id_pair $$$
            relation_info['class_id_pair'] = (int(detections.class_id[idx_i]), int(detections.class_id[idx_j]))
            relation_info['class_name_pair'] = (class_i, class_j)

            relations.append(relation_info)

    graph_dict['relations'] = relations

    return clip_embeds, seg_map, graph_dict

def create(args, img_folder, save_folder, sam_model):
    """
    处理图像文件夹中的所有图像，生成并保存特征和场景图数据
    :param args: 参数对象
    :param img_folder: 图像文件夹路径
    :param save_folder: 保存结果的文件夹路径
    :param sam_model: SAM 模型实例
    """
    data_list = os.listdir(img_folder)
    data_list.sort()
    assert len(data_list) > 0, "image_list must be provided to generate features"
    timer = 0
    embed_size=512
    seg_maps = []
    total_lengths = []
    # 预分配内存 
    image = cv2.imread(os.path.join(img_folder, data_list[0]))  
    global resolution
    resolution = (image.shape[1], image.shape[0])   
    resolution = (800,800) 
    img_embeds = torch.zeros((len(data_list), 100, embed_size)) # 为每张照片预分配100个区域的embedding空间，embedding大小为512
    seg_maps = torch.zeros((len(data_list), 4, resolution[1], resolution[0]))       # 为每张照片预分配4个层级的seg_map，大小为800x800
    llava_chat = LLaVaChat(args.llava_ckpt)                     # 初始化LLaVa聊天模型
    classes_set = set()
    global_class_counts = {} 
    global_class = set()
    # mask_generator.predictor.model
    number_save = 0
    for i, data_path in tqdm(enumerate(data_list), desc="Embedding images", leave=False): 
        # if number_save >= 10:
        #     break
        timer += 1
        torch.cuda.empty_cache() # 每处理一张图片就清理一次显存，防止显存溢出
        image_path = os.path.join(img_folder, data_path)

        # 构建单张图像的场景图
        # Updated to pass sam_model and args
        img_embed, seg_map, graph_dict = graph_construct(image_path, sam_predictor, sam_model, llava_chat, classes_set, global_class_counts,global_class, args) 
        number_save += 1

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)

        # 如果特征数量超过预分配的空间，进行扩容
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(data_list), pad, embed_size))
            ], dim=1)
        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
      
        img_embeds[i, :total_length] = img_embed
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):  
            if j == 0:  
                seg_map_tensor.append(torch.from_numpy(v))  
                continue  
            level_idx = list(seg_map.keys()).index(k)  
            assert v.max() == lengths[level_idx] - 1  
            # 调整分割图索引，使其在所有层级中唯一
            v[v != -1] += lengths_cumsum[level_idx-1]  
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map
      
        # 保存每个图像的 img_embed, seg_map和rel_info
        save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(image_path))[0])

        # 确保 seg_map 的最大值与长度一致
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i],
            'graph': graph_dict
        }

        sava_numpy(save_path, curr)

        # Explicit memory cleanup to prevent OOM
        # 显式清理内存以防止显存溢出
        del img_embed, seg_map, graph_dict, curr, seg_map_tensor, lengths, lengths_cumsum
        gc.collect()
        torch.cuda.empty_cache()
        
        # 如果存在 predictor_sam 全局变量且有 reset_predictor 方法，则重置
        if 'predictor_sam' in globals() and hasattr(predictor_sam, 'reset_predictor'):
            try:
                predictor_sam.reset_predictor()
            except:
                pass
    
    # Save global class counts
    dataset_path = "/home/zhengwu/Desktop/3D-Gaussian/GaussianGraph/dataset"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    global_counts_path = os.path.join(dataset_path, "global_class.json")
    try:
        with open(global_counts_path, 'w') as f:
            json.dump(list(global_class), f)
        print(f"Saved global class counts to {global_counts_path}")
    except Exception as e:
        print(f"Error saving global class counts: {e}")

def sava_numpy(save_path, data):
    """
    保存数据为 numpy 和 json 格式
    """
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    save_path_r = save_path + '_r.json'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())
    with open(save_path_r, 'w') as f:
        json.dump(data['graph'], f)

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./submodules/segment_anything/sam2/configs/sam2.1/sam2.1_hiera_l")
    parser.add_argument('--sam_ckpt', default="./submodules/segment_anything/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--dataset_path', type=str, default="./scannet/scans/scene0000_00/")
    parser.add_argument('--gsa_config', default="./submodules/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--gsa_ckpt', type=str, default="./submodules/groundingdino/groundingdino_swint_ogc.pth")
    parser.add_argument('--llava_ckpt', type=str, default="./submodules/llava/llava-next/llava_1.6")
    parser.add_argument("--box_threshold", type=float, default=0.2)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    parser.add_argument("--nms_threshold", type=float, default=0.2)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--skip_relations', action='store_true', help='Skip relationship extraction for speed')
    parser.add_argument('--skip_captions', action='store_true', help='Skip detailed object captions') 
    parser.add_argument('--use_known_classes', action='store_true', help='Use known classes for detection')
    parser.add_argument('--skip_background', action='store_true', help='Skip background class in classes')
    parser.add_argument('--relation_threshold', type=float, default=0.2, help='Threshold for object distance to consider relation (fraction of image diagonal)')
    parser.add_argument('--max_relations', type=int, default=50, help='Maximum number of relations to extract per image')
    parser.add_argument('--fast_sam', action='store_true', help='Use faster SAM settings')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path

    # 判断路径是否存在
    if os.path.exists(os.path.join(dataset_path, 'color')):
        img_folder = os.path.join(dataset_path, 'color')
    elif os.path.exists(os.path.join(dataset_path, 'images')):
        img_folder = os.path.join(dataset_path, 'images')
    else:
        raise ValueError('Image folder not found')

    # 初始化模型
    # 1. OpenCLIP 模型: 用于特征提取
    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    
    # 2. GroundingDINO 模型: 用于开放集目标检测
    grounding_dino_model = Model(model_config_path=args.gsa_config, model_checkpoint_path=args.gsa_ckpt, device=args.device)
    
    # 3. SAM2 模型: 用于分割
    # build_sam2: 加载模型结构和权重
    sam = build_sam2(args.config, args.sam_ckpt, args.device, apply_postprocessing=False)
    
    # SAM2ImagePredictor: 用于基于 Prompt (点/框) 的分割
    predictor_sam = SAM2ImagePredictor(sam_model=sam)
    
    # mask_generator = SAM2AutomaticMaskGenerator(model=sam)
    
    # Initialize generators for different scales
    # 初始化不同尺度的 SAM 自动掩码生成器
    # 移动到 sam_encoder 内部动态初始化，避免显存累积
    # points_per_side: 采样点数，越多越精细，但速度越慢
    # points_per_batch: 并行处理的点数，显存允许的情况下可以调大
    # points_s = 32 if args.fast_sam else 64
    # mask_generator_s = SAM2AutomaticMaskGenerator(model=sam, points_per_side=points_s, points_per_batch=8)
    # mask_generator_m = SAM2AutomaticMaskGenerator(model=sam, points_per_side=32, points_per_batch=8)
    # mask_generator_l = SAM2AutomaticMaskGenerator(model=sam, points_per_side=16, points_per_batch=8)
    # mask_generator = mask_generator_m

    WARNED = False

    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    create(args, img_folder, save_folder, sam)
