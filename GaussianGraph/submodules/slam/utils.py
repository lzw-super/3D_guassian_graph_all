#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SLAM系统工具函数模块

该模块提供了SLAM（同步定位与地图构建）系统中用于点云处理、对象管理、空间计算等的核心工具函数。
主要功能包括：
- 类别颜色生成与管理
- 点云创建、去噪和处理
- 边界框计算
- 对象融合与匹配
- 检测列表转换与变换
- 目标过滤与管理
"""

from collections import Counter  # 用于计数统计
import copy  # 用于深拷贝对象
import json  # 用于JSON文件读写
import cv2  # 计算机视觉库，用于图像处理

import numpy as np  # 数值计算库
from omegaconf import DictConfig  # 配置管理
import open3d as o3d  # 3D点云处理库
import torch  # PyTorch深度学习框架

import torch.nn.functional as F  # PyTorch函数库

import faiss  # 高效相似性搜索库

from conceptgraph.utils.general_utils import to_tensor, to_numpy, Timer  # 通用工具函数
from conceptgraph.slam.slam_classes import MapObjectList, DetectionList  # SLAM类定义

from conceptgraph.utils.ious import compute_3d_iou, compute_3d_iou_accuracte_batch, mask_subtract_contained, compute_iou_batch
from conceptgraph.dataset.datasets_common import from_intrinsics_matrix


def get_classes_colors(classes):
    """
    为每个类别生成随机RGB颜色映射
    
    Args:
        classes: 类别名称列表
        
    Returns:
        class_colors: 字典，键为类别索引，值为对应的RGB颜色元组（范围0-1）
    """
    class_colors = {}

    # 为每个类别生成随机颜色
    for class_idx, class_name in enumerate(classes):
        # 生成0到255之间的随机RGB值并归一化到0-1
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0

        # 将RGB值作为元组分配给字典中的类别
        class_colors[class_idx] = (r, g, b)

    # 为背景类别(-1)设置黑色
    class_colors[-1] = (0, 0, 0)

    return class_colors


def create_or_load_colors(cfg, filename="gsa_classes_tag2text"):
    """
    创建或加载类别颜色映射文件
    
    Args:
        cfg: 配置对象，包含数据集根目录和场景ID
        filename: 类别文件名（不含扩展名）
        
    Returns:
        classes: 类别列表
        class_colors: 类别颜色字典
    """
    filename = filename  # +'_lighthqsam'
    # 获取类别，应该在制作数据集时已保存
    classes_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}.json"
    classes = None
    with open(classes_fp, "r") as f:
        classes = json.load(f)
    
    # 创建类别颜色，或加载已存在的颜色文件
    class_colors = None
    class_colors_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}_colors.json"
    if class_colors_fp.exists():
        with open(class_colors_fp, "r") as f:
            class_colors = json.load(f)
        print("Loaded class colors from ", class_colors_fp)
    else:
        class_colors = get_classes_colors(classes)
        # 将键转换为字符串以符合JSON格式要求
        class_colors = {str(k): v for k, v in class_colors.items()}
        with open(class_colors_fp, "w") as f:
            json.dump(class_colors, f)
        print("Saved class colors to ", class_colors_fp)
    return classes, class_colors


def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None) -> o3d.geometry.PointCloud:
    """
    从深度图和掩码创建点云对象
    
    Args:
        depth_array: 深度图像数组
        mask: 二值掩码，指示对象区域
        cam_K: 相机内参矩阵
        image: RGB图像，用于着色
        obj_color: 可选，自定义对象颜色
        
    Returns:
        pcd: Open3D点云对象
    """
    # 从相机内参矩阵提取焦距和主点坐标
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    
    # 同时移除无效深度值的点
    mask = np.logical_and(mask, depth_array > 0)

    # 如果掩码中没有有效点，返回空点云
    if mask.sum() == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd
        
    # 获取深度图尺寸并创建网格坐标
    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # 应用掩码，仅在有效点上进行反投影
    masked_depth = depth_array[mask]  # (N, )
    u = u[mask]  # (N, )
    v = v[mask]  # (N, )

    # 转换为3D坐标
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fy
    z = masked_depth

    # 将x, y, z坐标堆叠成3D点云
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    
    # 对点进行微小扰动以避免共线性
    points += np.random.normal(0, 4e-3, points.shape)

    # 根据是否提供自定义颜色选择着色方式
    if obj_color is None:  # 使用RGB图像着色
        colors = image[mask] / 255.0
    else:  # 使用组ID着色
        colors = np.full(points.shape, obj_color)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    """
    使用DBSCAN聚类算法对点云进行去噪
    
    Args:
        pcd: 输入点云
        eps: DBSCAN邻域半径参数
        min_points: DBSCAN最小点数阈值
        
    Returns:
        pcd: 去噪后的点云（最大聚类部分）
    """
    ### 通过聚类去除噪声
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # 转换为numpy数组
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # 统计所有标签的数量
    counter = Counter(pcd_clusters)

    # 移除噪声标签（-1表示噪声点）
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # 找到最大聚类的标签
        most_common_label, _ = counter.most_common(1)[0]
        
        # 创建最大聚类点的掩码
        largest_mask = pcd_clusters == most_common_label

        # 应用掩码
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # 如果最大聚类太小，返回原始点云
        if len(largest_cluster_points) < 5:
            return pcd

        # 创建新的点云对象
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd


def process_pcd(pcd, cfg, run_dbscan=True):
    """
    对点云进行预处理，包括降采样和去噪
    
    Args:
        pcd: 输入点云
        cfg: 配置对象，包含处理参数
        run_dbscan: 是否执行DBSCAN去噪
        
    Returns:
        pcd: 处理后的点云
    """
    # 体素降采样
    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_voxel_size)
        
    # 根据配置执行DBSCAN去噪
    if cfg.dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=cfg.dbscan_eps, 
            min_points=cfg.dbscan_min_points
        )
        
    return pcd


def get_bounding_box(cfg, pcd):
    """
    根据配置获取点云的边界框
    
    Args:
        cfg: 配置对象，包含空间相似性类型
        pcd: 输入点云
        
    Returns:
        bbox: 点云的边界框（定向或轴对齐）
    """
    # 如果需要精确的空间相似性计算且点云点数足够，使用定向边界框
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        # 否则使用轴对齐边界框
        return pcd.get_axis_aligned_bounding_box()


def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    """
    将新对象（obj2）合并到旧对象（obj1）中
    此操作是原地进行的
    
    Args:
        cfg: 配置对象
        obj1: 目标对象（将被更新）
        obj2: 源对象（将被合并）
        run_dbscan: 是否对点云执行DBSCAN去噪
        
    Returns:
        obj1: 合并后的对象
    """
    # 获取两个对象的检测次数
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    # 遍历obj1的所有键进行合并
    for k in obj1.keys():
        if k in ['caption']:
            # 合并两个字典并调整第二个字典的键
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft"]:
            # 处理列表和整数类型
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                # 保持初始实例颜色
                obj1[k] = obj1[k]
            else:
                # 未来可能需要处理其他类型
                raise NotImplementedError
        else:  # pcd, bbox, clip_ft, text_ft在下面单独处理
            continue
            
    # 合并点云和边界框
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan)
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]  # 设置边界框颜色为绿色
    
    # 合并CLIP特征（视觉特征），按检测次数加权平均
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
                       obj2['clip_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)  # 归一化

    # 合并文本特征，按检测次数加权平均
    obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg.device)
    obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg.device)
    obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
                       obj2['text_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)  # 归一化
    
    return obj1


def compute_overlap_matrix(cfg, objects: MapObjectList):
    """
    计算对象之间的点云重叠矩阵
    重叠度定义为：对象i中点到对象j中最近点的距离小于阈值的点比例
    
    Args:
        cfg: 配置对象，包含阈值参数
        objects: MapObjectList对象列表
        
    Returns:
        overlap_matrix: n x n的numpy数组，表示对象间重叠程度
    """
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # 将点云转换为numpy数组并创建FAISS索引以进行高效搜索
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    
    # 将点添加到对应的FAISS索引中
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # 计算两两重叠度
    for i in range(n):
        for j in range(n):
            if i != j:  # 跳过对角线元素
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                
                # 如果边界框完全不重叠，跳过计算（节省计算资源）
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                
                # 使用FAISS搜索最近邻点
                D, I = indices[j].search(point_arrays[i], 1)

                # 计算距离小于阈值的点数量
                overlap = (D < cfg.downsample_voxel_size ** 2).sum()  # D是平方距离

                # 计算重叠比例
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix


def compute_overlap_matrix_2set(cfg, objects_map: MapObjectList, objects_new: DetectionList) -> np.ndarray:
    """
    计算两组对象之间的点云重叠矩阵
    用于比较地图中已存在的对象和新检测的对象
    
    Args:
        cfg: 配置对象，包含阈值参数
        objects_map: 地图中已存在的对象
        objects_new: 新检测的对象
        
    Returns:
        overlap_matrix: m x n的numpy数组，表示两组对象间的重叠程度
    """
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # 将地图中对象的点云转换为numpy数组并创建FAISS索引
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map]  # m个数组
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map]  # m个索引
    
    # 将点添加到对应的FAISS索引中
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    # 获取新检测对象的点云
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new]  # n个数组
        
    # 获取两组对象的边界框并计算IOU
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new)  # (m, n)
    except ValueError:
        # 如果计算定向边界框IOU失败，使用轴对齐边界框
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects_map.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in objects_new.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new)  # (m, n)
            
    # 计算两两重叠度
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:  # 如果IOU过小，跳过
                continue
            
            # 在地图对象i中搜索新对象j的最近邻点
            D, I = indices[i].search(points_new[j], 1)

            # 计算距离小于阈值的点数量
            overlap = (D < cfg.downsample_voxel_size ** 2).sum()  # D是平方距离

            # 计算重叠比例
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix


def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray):
    """
    合并重叠的对象
    基于重叠度、视觉相似性和文本相似性进行判断
    
    Args:
        cfg: 配置对象，包含合并阈值
        objects: MapObjectList对象列表
        overlap_matrix: 对象间重叠矩阵
        
    Returns:
        objects: 合并后的对象列表
    """
    # 找到所有非零重叠的位置
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    # 按重叠度降序排序
    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    # 跟踪哪些对象需要保留
    kept_objects = np.ones(len(objects), dtype=bool)
    
    # 遍历排序后的重叠对
    for i, j, ratio in zip(x, y, overlap_ratio):
        # 计算视觉相似性（CLIP特征的余弦相似度）
        visual_sim = F.cosine_similarity(
            to_tensor(objects[i]['clip_ft']),
            to_tensor(objects[j]['clip_ft']),
            dim=0
        )
        # 计算文本相似性（文本特征的余弦相似度）
        text_sim = F.cosine_similarity(
            to_tensor(objects[i]['text_ft']),
            to_tensor(objects[j]['text_ft']),
            dim=0
        )
        
        # 如果重叠度和相似度都超过阈值，则合并对象
        if ratio > cfg.merge_overlap_thresh:
            if visual_sim > cfg.merge_visual_sim_thresh and \
                text_sim > cfg.merge_text_sim_thresh:
                if kept_objects[j]:
                    # 将对象i合并到对象j中
                    objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
                    kept_objects[i] = False
        else:
            # 由于已排序，后面的重叠度更小，直接中断循环
            break
    
    # 移除已合并的对象
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    
    return objects


def denoise_objects(cfg, objects: MapObjectList):
    """
    对对象列表中的每个对象进行去噪处理
    
    Args:
        cfg: 配置对象
        objects: MapObjectList对象列表
        
    Returns:
        objects: 去噪后的对象列表
    """
    for i in range(len(objects)):
        # 保存原始点云以便在处理后点太少时恢复
        og_object_pcd = objects[i]['pcd']
        # 对点云进行去噪
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], cfg, run_dbscan=True)
        
        # 如果处理后点太少，恢复原始点云
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
            
        # 更新边界框
        objects[i]['bbox'] = get_bounding_box(cfg, objects[i]['pcd'])
        objects[i]['bbox'].color = [0,1,0]  # 设置为绿色
        
    return objects


def filter_objects(cfg, objects: MapObjectList):
    """
    过滤对象列表，移除点数太少或检测次数太少的对象
    
    Args:
        cfg: 配置对象，包含过滤阈值
        objects: MapObjectList对象列表
        
    Returns:
        objects: 过滤后的对象列表
    """
    # 打印过滤前后的对象数量
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        # 保留点数和检测次数都超过阈值的对象
        if len(obj['pcd'].points) >= cfg.obj_min_points and obj['num_detections'] >= cfg.obj_min_detections:
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))
    
    return objects


def merge_objects(cfg, objects: MapObjectList):
    """
    合并重叠的对象
    如果配置的合并阈值大于0，则执行合并
    
    Args:
        cfg: 配置对象，包含合并阈值
        objects: MapObjectList对象列表
        
    Returns:
        objects: 合并后的对象列表
    """
    if cfg.merge_overlap_thresh > 0:
        # 计算对象间重叠矩阵
        overlap_matrix = compute_overlap_matrix(cfg, objects)
        print("Before merging:", len(objects))
        # 合并重叠对象
        objects = merge_overlap_objects(cfg, objects, overlap_matrix)
        print("After merging:", len(objects))
    
    return objects


def filter_gobs(
    cfg: DictConfig,
    gobs: dict,
    image: np.ndarray,
    BG_CLASSES = ["wall", "floor", "ceiling"],
):
    """
    过滤检测到的对象组（gobs）
    根据掩码大小、类别、边界框大小和置信度等条件进行过滤
    
    Args:
        cfg: 配置对象
        gobs: 检测对象组字典
        image: 输入图像
        BG_CLASSES: 背景类别列表
        
    Returns:
        gobs: 过滤后的对象组
    """
    # 如果没有检测到任何对象，直接返回
    if len(gobs['xyxy']) == 0:
        return gobs
    
    # 过滤对象，保留满足条件的对象索引
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        
        # 跳过太小的掩码
        if gobs['mask'][mask_idx].sum() < max(cfg.mask_area_threshold, 10):
            continue
        
        # 跳过背景类别
        if cfg.skip_bg and class_name in BG_CLASSES:
            continue
        
        # 跳过太大的非背景框
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if bbox_area > cfg.max_bbox_area_ratio * image_area:
                continue
        
        # 跳过低置信度掩码
        if gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < cfg.mask_conf_threshold:
                continue
        
        idx_to_keep.append(mask_idx)
    
    # 根据保留的索引更新所有字段
    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes":  # 跳过字符串和类别列表
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")
    
    return gobs


def resize_gobs(
    gobs,
    image
):
    """
    调整对象组的掩码大小以匹配输入图像尺寸
    
    Args:
        gobs: 检测对象组字典
        image: 目标图像
        
    Returns:
        gobs: 调整大小后的对象组
    """
    n_masks = len(gobs['xyxy'])
    new_mask = []
    
    for mask_idx in range(n_masks):
        # 调整掩码大小以匹配图像尺寸
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
            # 重新缩放边界框坐标
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            x1 = round(x1 * image.shape[1] / mask.shape[1])
            y1 = round(y1 * image.shape[0] / mask.shape[0])
            x2 = round(x2 * image.shape[1] / mask.shape[1])
            y2 = round(y2 * image.shape[0] / mask.shape[0])
            gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]
            
            # 调整掩码大小
            mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            new_mask.append(mask)

    # 更新掩码数组
    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)
        
    return gobs


def gobs_to_detection_list(
    cfg,
    image,
    depth_array,
    cam_K,
    idx,
    gobs,
    trans_pose = None,
    class_names = None,
    BG_CLASSES = ["wall", "floor", "ceiling"],
    color_path = None,
):
    """
    将检测对象组（gobs）转换为DetectionList对象
    所有对象仍在相机坐标系中
    
    Args:
        cfg: 配置对象
        image: RGB图像
        depth_array: 深度图像
        cam_K: 相机内参矩阵
        idx: 图像索引
        gobs: 检测对象组字典
        trans_pose: 可选的变换矩阵
        class_names: 全局类别名称列表
        BG_CLASSES: 背景类别列表
        color_path: 颜色图像路径
        
    Returns:
        fg_detection_list: 前景对象检测列表
        bg_detection_list: 背景对象检测列表
    """
    # 初始化前景和背景检测列表
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()
    
    # 调整掩码大小并过滤对象
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES)
    
    # 如果没有检测到对象，返回空列表
    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list
    
    # 计算所有检测之间的包含关系，并从背景对象中减去前景对象
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    
    # 处理每个掩码
    n_masks = len(gobs['xyxy'])
    for mask_idx in range(n_masks):
        local_class_id = gobs['class_id'][mask_idx]
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        # 获取全局类别ID
        global_class_id = -1 if class_names is None else class_names.index(class_name)
        
        # 创建点云并着色
        camera_object_pcd = create_object_pcd(
            depth_array,
            mask,
            cam_K,
            image,
            obj_color = None
        )
        
        # 确保点云包含足够的点
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5):
            continue
        
        # 应用变换矩阵（如果提供）
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # 获取最大聚类，过滤噪声
        global_object_pcd = process_pcd(global_object_pcd, cfg)
        
        # 计算边界框
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0,1,0]
        
        # 跳过体积过小的对象
        if pcd_bbox.volume() < 1e-6:
            continue
        
        # 将检测视为3D对象，存储足够的信息以恢复检测
        detected_object = {
            'image_idx': [idx],                           # 图像索引
            'mask_idx': [mask_idx],                       # 掩码/检测索引
            'color_path': [color_path],                   # RGB图像路径
            'class_name': [class_name],                   # 检测的类别名称
            'class_id': [global_class_id],                # 全局类别ID
            'num_detections': 1,                          # 该对象中的检测次数
            'mask': [mask],                               # 掩码
            'xyxy': [gobs['xyxy'][mask_idx]],             # 边界框坐标
            'conf': [gobs['confidence'][mask_idx]],       # 置信度
            'n_points': [len(global_object_pcd.points)],  # 点云点数
            'pixel_area': [mask.sum()],                   # 像素面积
            'contain_number': [None],                     # 包含关系数量（后续计算）
            "inst_color": np.random.rand(3),              # 实例随机颜色
            'is_background': class_name in BG_CLASSES,    # 是否为背景
            
            # 3D对象信息
            'pcd': global_object_pcd,                     # 点云
            'bbox': pcd_bbox,                             # 边界框
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),  # 视觉特征
            'text_ft': to_tensor(gobs['text_feats'][mask_idx]),   # 文本特征
        }
        
        # 根据类别将对象添加到前景或背景列表
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)
    
    return fg_detection_list, bg_detection_list


def transform_detection_list(
    detection_list: DetectionList,
    transform: torch.Tensor,
    deepcopy = False,
):
    """
    通过给定的变换矩阵变换检测列表
    
    Args:
        detection_list: DetectionList对象
        transform: 4x4变换矩阵
        deepcopy: 是否深拷贝检测列表
        
    Returns:
        transformed_detection_list: 变换后的DetectionList对象
    """
    # 将变换矩阵转换为numpy格式
    transform = to_numpy(transform)
    
    # 如果需要，深拷贝检测列表
    if deepcopy:
        detection_list = copy.deepcopy(detection_list)
    
    # 对每个检测对象应用变换
    for i in range(len(detection_list)):
        # 变换点云
        detection_list[i]['pcd'] = detection_list[i]['pcd'].transform(transform)
        # 变换边界框（先旋转后平移）
        detection_list[i]['bbox'] = detection_list[i]['bbox'].rotate(transform[:3, :3], center=(0, 0, 0))
        detection_list[i]['bbox'] = detection_list[i]['bbox'].translate(transform[:3, 3])
        # 也可以选择直接重新计算边界框，但上述方法更高效
        # detection_list[i]['bbox'] = detection_list[i]['pcd'].get_oriented_bounding_box(robust=True)
    
    return detection_list


def compute_2d_box_contained_batch(bbox: torch.Tensor, thresh:float=0.95) -> torch.Tensor:
    """
    对于每个边界框，计算有多少其他边界框包含它
    
    首先计算每对边界框之间的交集面积，然后对每个边界框，统计有多少个边界框的交集面积
    大于其自身面积的thresh比例
    
    Args:
        bbox: 形状为(N, 4)的张量，格式为(x1, y1, x2, y2)
        thresh: 浮点数，判断包含关系的阈值
    
    Returns:
        torch.Tensor: 形状为(N,)的张量，表示每个边界框被包含的次数
    """
    N = bbox.shape[0]

    # 计算每个边界框的面积
    areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    # 计算交集边界框的左上角和右下角坐标
    lt = torch.max(bbox[:, None, :2], bbox[:, :2])  # 左上角点 (N, N, 2)
    rb = torch.min(bbox[:, None, 2:], bbox[:, 2:])  # 右下角点 (N, N, 2)

    # 计算交集的尺寸，将负值设为0
    inter = (rb - lt).clamp(min=0)  # 交集尺寸 (dx, dy) (N, N, 2)

    # 计算交集面积
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    # 统计满足包含条件的边界框数量
    mask = inter_areas > (areas * thresh).unsqueeze(1)  # (N, N)
    count = mask.sum(dim=1) - 1  # 排除自身 (N,)

    return count