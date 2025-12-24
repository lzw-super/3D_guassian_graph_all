# -*- coding: utf-8 -*-
"""
SLAM映射模块 - mapping.py

该模块实现了SLAM（同时定位与地图构建）系统中的关键映射功能，
主要负责计算检测对象与地图对象之间的空间和视觉相似度，
并基于这些相似度将新检测到的对象合并到现有地图中。

核心功能包括：
1. 空间相似度计算（基于IoU、GIoU等）
2. 视觉相似度计算（基于CLIP特征的余弦相似度）
3. 相似度聚合（空间与视觉相似度的融合）
4. 检测对象与地图对象的合并
"""
import torch  # 导入PyTorch库，用于张量计算和深度学习操作
import torch.nn.functional as F  # 导入PyTorch的函数式API，提供各种神经网络函数

# 从SLAM类模块导入地图对象列表和检测列表类
from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
# 导入计时器工具，用于性能测量
from conceptgraph.utils.general_utils import Timer
# 导入各种IoU（交并比）计算函数
from conceptgraph.utils.ious import (
    compute_iou_batch,  # 批量计算2D IoU
    compute_giou_batch,  # 批量计算2D GIoU
    compute_3d_iou_accuracte_batch,  # 批量计算精确的3D IoU
    compute_3d_giou_accurate_batch,  # 批量计算精确的3D GIoU
)
# 导入SLAM工具函数
from conceptgraph.slam.utils import (
    merge_obj2_into_obj1,  # 将对象2合并到对象1中
    compute_overlap_matrix_2set  # 计算两个集合之间的重叠矩阵
)

def compute_spatial_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    """
    计算检测对象与地图对象之间的空间相似度
    
    该函数根据配置选择不同的空间相似度计算方法，包括各种IoU变体和重叠计算
    用于衡量检测到的对象与地图中现有对象之间的空间重叠程度
    
    Args:
        cfg: 配置对象，包含空间相似度计算类型等参数
        detection_list: 检测对象列表，包含M个检测结果
        objects: 地图对象列表，包含N个已存在的地图对象
    
    Returns:
        torch.Tensor: 形状为[M, N]的张量，表示每个检测与每个地图对象之间的空间相似度
    """
    # 获取所有检测对象的边界框并堆叠成张量
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    # 获取所有地图对象的边界框并堆叠成张量
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    # 根据配置选择不同的空间相似度计算方法
    if cfg.spatial_sim_type == "iou":
        # 使用2D IoU（交并比）计算空间相似度
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou":
        # 使用2D GIoU（广义交并比）计算空间相似度，考虑了边界框的位置关系
        spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "iou_accurate":
        # 使用精确的3D IoU计算空间相似度，适用于3D场景
        spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou_accurate":
        # 使用精确的3D GIoU计算空间相似度
        spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "overlap":
        # 使用重叠矩阵计算空间相似度
        spatial_sim = compute_overlap_matrix_2set(cfg, objects, detection_list)
        # 将numpy数组转换为PyTorch张量并转置，使其形状为[M, N]
        spatial_sim = torch.from_numpy(spatial_sim).T
    else:
        # 如果配置了无效的空间相似度类型，抛出异常
        raise ValueError(f"无效的空间相似度类型: {cfg.spatial_sim_type}")
    
    return spatial_sim

def compute_visual_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    """
    计算检测对象与地图对象之间的视觉相似度
    
    该函数使用CLIP特征计算检测到的对象与地图中现有对象之间的视觉相似度，
    通过余弦相似度衡量特征向量之间的相似程度
    
    Args:
        cfg: 配置对象
        detection_list: 检测对象列表，包含M个检测结果
        objects: 地图对象列表，包含N个已存在的地图对象
    
    Returns:
        torch.Tensor: 形状为[M, N]的张量，表示每个检测与每个地图对象之间的视觉相似度
    """
    # 获取所有检测对象的CLIP特征并堆叠成张量，形状为[M, D]，其中D是特征维度
    det_fts = detection_list.get_stacked_values_torch('clip_ft')  # (M, D)
    # 获取所有地图对象的CLIP特征并堆叠成张量，形状为[N, D]
    obj_fts = objects.get_stacked_values_torch('clip_ft')  # (N, D)

    # 在最后一维添加一个维度，形状变为[M, D, 1]
    det_fts = det_fts.unsqueeze(-1)  # (M, D, 1)
    # 转置并在第一维添加一个维度，形状变为[1, D, N]
    obj_fts = obj_fts.T.unsqueeze(0)  # (1, D, N)
    
    # 使用PyTorch的余弦相似度函数计算相似度，在特征维度(dim=1)上计算
    # 利用广播机制，最终得到形状为[M, N]的相似度矩阵
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1)  # (M, N)
    
    return visual_sim

def aggregate_similarities(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    """
    聚合空间相似度和视觉相似度为单一的综合相似度分数
    
    该函数将空间相似度和视觉相似度进行加权融合，生成一个综合的相似度矩阵，
    用于后续的对象匹配和合并决策
    
    Args:
        cfg: 配置对象，包含匹配方法和物理偏差参数
        spatial_sim: 形状为[M, N]的空间相似度张量
        visual_sim: 形状为[M, N]的视觉相似度张量
    
    Returns:
        torch.Tensor: 形状为[M, N]的综合相似度张量
    """
    # 根据配置选择相似度聚合方法
    if cfg.match_method == "sim_sum":
        # 使用加权求和的方式聚合空间和视觉相似度
        # cfg.phys_bias参数控制空间相似度的权重，范围应该在[-1, 1]之间
        # 当phys_bias为正时，增加空间相似度的权重；为负时，增加视觉相似度的权重
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim  # (M, N)
    else:
        # 如果配置了未知的匹配方法，抛出异常
        raise ValueError(f"未知的匹配方法: {cfg.match_method}")
    
    return sims

def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    """
    将检测到的对象合并到地图对象列表中
    
    该函数遍历所有检测到的对象，根据综合相似度决定是将其添加为新对象，
    还是合并到现有的最相似的地图对象中
    
    Args:
        cfg: 配置对象
        detection_list: 检测对象列表，包含M个检测结果
        objects: 地图对象列表，包含N个已存在的地图对象
        agg_sim: 形状为[M, N]的综合相似度张量
    
    Returns:
        MapObjectList: 更新后的地图对象列表
    """
    # 遍历所有检测对象
    for i in range(agg_sim.shape[0]):
        # 检查当前检测是否匹配到任何地图对象
        # 如果最大相似度为负无穷（表示没有匹配），则将其添加为新对象
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # 否则，将其合并到最相似的现有地图对象中
        else:
            # 找到最相似的地图对象的索引
            j = agg_sim[i].argmax()
            # 获取当前检测对象和匹配的地图对象
            matched_det = detection_list[i]
            matched_obj = objects[j]
            # 将检测对象合并到地图对象中，不运行DBSCAN聚类
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            # 更新地图对象列表中的对象
            objects[j] = merged_obj
            
    return objects