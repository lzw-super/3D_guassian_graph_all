
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLAM 系统中的核心类定义模块

此模块定义了 SLAM (Simultaneous Localization and Mapping) 系统中用于管理检测对象和地图对象的两个核心类：
1. DetectionList - 用于管理场景中的检测对象集合
2. MapObjectList - 继承自 DetectionList，专门用于管理地图中的持久化对象

这些类提供了丰富的方法来处理点云数据、边界框、特征向量以及对象之间的相似度计算等功能。
"""
from collections.abc import Iterable
import copy
import matplotlib  # 用于生成颜色映射
import torch  # PyTorch 库，用于张量操作
import torch.nn.functional as F  # PyTorch 功能模块，用于计算余弦相似度等
import numpy as np  # NumPy 库，用于数组操作
import open3d as o3d  # Open3D 库，用于点云和 3D 几何处理

from conceptgraph.utils.general_utils import to_numpy, to_tensor  # 自定义工具函数，用于在 numpy 和 tensor 之间转换

class DetectionList(list):
    """
    检测对象列表类，继承自 Python 的内置 list 类
    
    用于管理和操作场景中检测到的对象集合，提供了丰富的方法来获取和处理对象属性、
    进行对象过滤以及可视化相关操作。
    """
    def get_values(self, key, idx:int=None):
        """
        获取列表中所有检测对象的指定键的值
        
        参数:
            key (str): 要获取的属性名称
            idx (int, optional): 如果属性值是列表或数组，可指定索引获取特定元素
        
        返回:
            list: 包含所有检测对象指定属性值的列表
        """
        if idx is None:
            # 如果没有指定索引，直接获取所有对象的指定键的值
            return [detection[key] for detection in self]
        else:
            # 如果指定了索引，获取所有对象指定键的值的特定索引元素
            return [detection[key][idx] for detection in self]
    
    def get_stacked_values_torch(self, key, idx:int=None):
        """
        获取列表中所有检测对象的指定键的值，并将其转换为 PyTorch 张量后堆叠
        
        参数:
            key (str): 要获取的属性名称
            idx (int, optional): 如果属性值是列表或数组，可指定索引获取特定元素
        
        返回:
            torch.Tensor: 堆叠后的张量，形状为 [N, ...]，N 为检测对象数量
        """
        values = []
        for detection in self:
            v = detection[key]  # 获取当前对象的指定属性值
            if idx is not None:
                v = v[idx]  # 如果指定了索引，获取特定元素
            
            # 处理 Open3D 边界框类型，将其转换为 numpy 数组
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            
            # 将 numpy 数组转换为 PyTorch 张量
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            
            values.append(v)
        
        # 沿维度 0 堆叠所有值，形成一个张量
        return torch.stack(values, dim=0)
    
    def get_stacked_values_numpy(self, key, idx:int=None):
        """
        获取列表中所有检测对象的指定键的值，并将其转换为 NumPy 数组后堆叠
        
        参数:
            key (str): 要获取的属性名称
            idx (int, optional): 如果属性值是列表或数组，可指定索引获取特定元素
        
        返回:
            np.ndarray: 堆叠后的 NumPy 数组
        """
        # 先获取 PyTorch 张量形式的值
        values = self.get_stacked_values_torch(key, idx)
        # 然后转换为 NumPy 数组
        return to_numpy(values)
    
    def __add__(self, other):
        """
        重载加法运算符，实现两个 DetectionList 的深拷贝合并
        
        参数:
            other (DetectionList or list): 要合并的另一个列表
        
        返回:
            DetectionList: 合并后的新列表
        """
        # 创建当前列表的深拷贝
        new_list = copy.deepcopy(self)
        # 将另一个列表的所有元素添加到新列表中
        new_list.extend(other)
        return new_list
    
    def __iadd__(self, other):
        """
        重载原地加法运算符，将另一个列表的元素原地添加到当前列表
        
        参数:
            other (DetectionList or list): 要添加的另一个列表
        
        返回:
            DetectionList: 添加后的当前列表
        """
        self.extend(other)
        return self
    
    def slice_by_indices(self, index: Iterable[int]):
        """
        根据索引集合获取当前列表的子列表
        
        参数:
            index (Iterable[int]): 要保留的元素索引集合
        
        返回:
            DetectionList: 包含指定索引元素的新列表
        """
        new_self = type(self)()  # 创建与当前类相同类型的新列表
        for i in index:
            new_self.append(self[i])  # 添加指定索引的元素
        return new_self
    
    def slice_by_mask(self, mask: Iterable[bool]):
        """
        根据掩码获取当前列表的子列表
        
        参数:
            mask (Iterable[bool]): 布尔掩码，True 表示保留对应索引的元素
        
        返回:
            DetectionList: 包含掩码为 True 的元素的新列表
        """
        new_self = type(self)()  # 创建与当前类相同类型的新列表
        for i, m in enumerate(mask):
            if m:  # 如果掩码为 True，则保留该元素
                new_self.append(self[i])
        return new_self
    
    def get_most_common_class(self) -> list[int]:
        """
        获取每个检测对象中最常见的类别 ID
        
        返回:
            list[int]: 每个检测对象最常见类别 ID 的列表
        """
        classes = []
        for d in self:
            # 获取当前对象的所有类别 ID 并转换为 numpy 数组
            values, counts = np.unique(np.asarray(d['class_id']), return_counts=True)
            # 找出出现次数最多的类别 ID
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes
    
    def color_by_most_common_classes(self, colors_dict: dict[str, list[float]], color_bbox: bool=True):
        """
        根据每个检测对象的最常见类别为其点云着色
        
        参数:
            colors_dict (dict[str, list[float]]): 类别 ID 到颜色值的映射字典，颜色值为 RGB 格式 [r, g, b]
            color_bbox (bool): 是否同时为边界框着色
        """
        # 获取每个检测对象的最常见类别
        classes = self.get_most_common_class()
        # 为每个检测对象的点云和可选的边界框着色
        for d, c in zip(self, classes):
            color = colors_dict[str(c)]  # 获取对应的颜色
            d['pcd'].paint_uniform_color(color)  # 为点云着色
            if color_bbox:
                d['bbox'].color = color  # 为边界框着色
                
    def color_by_instance(self):
        """
        根据实例为点云和边界框着色
        
        如果对象已有 'inst_color' 属性，则使用该颜色；否则生成唯一的颜色
        使用 'turbo' 颜色映射确保不同实例之间有明显的颜色区分
        """
        if len(self) == 0:
            # 如果列表为空，不进行任何操作
            return
        
        if "inst_color" in self[0]:
            # 如果第一个对象有 'inst_color' 属性，则使用现有颜色
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            # 否则使用 matplotlib 的 'turbo' 颜色映射生成不同的颜色
            cmap = matplotlib.colormaps.get_cmap("turbo")
            # 生成均匀分布在 [0, 1] 区间的颜色值
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            # 只取 RGB 三个通道的值（去掉 alpha 通道）
            instance_colors = instance_colors[:, :3]
            # 为每个检测对象分配一个唯一的颜色
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]
            
    
class MapObjectList(DetectionList):
    """
    地图对象列表类，继承自 DetectionList
    
    专门用于管理 SLAM 地图中的持久化对象，扩展了 DetectionList 的功能，
    增加了相似度计算和序列化相关的方法。
    """
    def compute_similarities(self, new_clip_ft):
        """
        计算新特征与地图中所有对象特征之间的相似度
        
        参数:
            new_clip_ft: 新的 CLIP 特征向量，可以是 numpy 数组或 PyTorch 张量，形状为 (D, )
        
        返回:
            torch.Tensor: 新特征与每个地图对象特征的余弦相似度，形状为 [N, ]
        """
        # 将输入特征转换为 PyTorch 张量（如果不是的话）
        new_clip_ft = to_tensor(new_clip_ft)
        
        # 获取地图中所有对象的 CLIP 特征，堆叠成张量
        clip_fts = self.get_stacked_values_torch('clip_ft')

        # 计算余弦相似度：先将新特征扩展为二维张量 [1, D]，然后与所有对象特征计算相似度
        similarities = F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)
        
        return similarities
    
    def to_serializable(self):
        """
        将地图对象列表转换为可序列化的格式，便于保存到文件
        
        将不可序列化的对象（如 Open3D 点云和边界框）转换为 numpy 数组格式
        
        返回:
            list: 包含可序列化对象字典的列表
        """
        s_obj_list = []
        for obj in self:
            # 创建对象字典的深拷贝，避免修改原始数据
            s_obj_dict = copy.deepcopy(obj)
            
            # 将 PyTorch 张量转换为 numpy 数组
            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])
            
            # 将 Open3D 点云转换为 numpy 数组表示
            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            # 将 Open3D 边界框转换为 numpy 数组表示（存储边界框的角点）
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            # 存储点云的颜色信息
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            
            # 删除不可序列化的 Open3D 对象
            del s_obj_dict['pcd']
            del s_obj_dict['bbox']
            
            s_obj_list.append(s_obj_dict)
            
        return s_obj_list
    
    def load_serializable(self, s_obj_list):
        """
        从可序列化格式加载地图对象列表
        
        将 numpy 数组转换回 Open3D 对象和 PyTorch 张量
        
        参数:
            s_obj_list (list): 包含可序列化对象字典的列表，通常是 to_serializable 方法的输出
        
        断言:
            要求当前 MapObjectList 为空，避免覆盖现有数据
        """
        # 确保当前列表为空
        assert len(self) == 0, '加载时 MapObjectList 应为空'
        
        for s_obj_dict in s_obj_list:
            # 创建对象字典的深拷贝
            new_obj = copy.deepcopy(s_obj_dict)
            
            # 将 numpy 数组转换回 PyTorch 张量
            new_obj['clip_ft'] = to_tensor(new_obj['clip_ft'])
            new_obj['text_ft'] = to_tensor(new_obj['text_ft'])
            
            # 重新创建 Open3D 点云对象
            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
            
            # 从角点创建定向边界框
            new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(new_obj['bbox_np']))
            
            # 设置边界框颜色（使用点云的第一个点的颜色）
            new_obj['bbox'].color = new_obj['pcd_color_np'][0]
            # 设置点云颜色
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])
            
            # 删除临时的 numpy 数组表示
            del new_obj['pcd_np']
            del new_obj['bbox_np']
            del new_obj['pcd_color_np']
            
            # 将重建的对象添加到列表中
            self.append(new_obj)