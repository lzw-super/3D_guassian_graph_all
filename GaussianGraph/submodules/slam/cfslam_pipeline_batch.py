''' 
cfslam_pipeline_batch.py

这个脚本用于将 Grounded SAM (GSA) 的检测结果建模到三维地图中。
主要假设与约定：
- 数据集（RGB + 深度 + 相机内参数 + 相机位姿）由 `get_dataset` 提供。
- 每帧的 GSA 检测（mask、类别、clip 特征等）已保存为压缩的 pickle 文件（.pkl.gz）。
- 外部依赖：Open3D、PyTorch、PIL、Hydra/OMEGACONF 等。

脚本功能概要：
- 逐帧加载 RGB/深度/位姿与 GSA 检测
- 将每个检测转换为 3D 点云（使用深度与内参投影）
- 使用空间相似度（spatial_sim）与视觉相似度（visual_sim）计算检测与地图中已有对象的相似性
- 根据相似性融合/合并检测到地图对象中（包括背景类的特殊处理）
- 可选：周期性去噪、过滤、合并对象；保存中间帧对象；渲染可视化视频；保存最终点云

文件中会对关键函数和步骤添加注释，便于理解与调试。
'''

# Standard library imports
import copy
from datetime import datetime
import os
from pathlib import Path
import gzip
import pickle

# Related third party imports
from PIL import Image
import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

import hydra
import omegaconf
from omegaconf import DictConfig

# Local application/library specific imports
from submodules.dataset.datasets_common import get_dataset
# from conceptgraph.utils.vis import OnlineObjectRenderer
from submodules.slam.utils import (
    compute_2d_box_contained_batch
)
# from submodules.utils.general_utils import to_tensor

from submodules.slam.slam_classes import MapObjectList, DetectionList
from submodules.slam.utils import (
    create_or_load_colors,
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
)
from submodules.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
) 

# 背景类别列表，这些类别会被单独处理并作为单个对象融合到地图中
BG_CLASSES = ["wall", "floor", "ceiling"]

# 禁用PyTorch的梯度计算，提高性能
torch.set_grad_enabled(False)

def compute_match_batch(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    根据空间相似度与视觉相似度计算检测与已有地图对象之间的关联矩阵（分配矩阵）。

    输入：
      - cfg: 配置对象，包含匹配方法、物理偏差权重、相似度阈值等参数
      - spatial_sim: MxN 张量，表示 M 个检测与 N 个对象之间的空间相似度
      - visual_sim:  MxN 张量，表示视觉（特征/语义）相似度

    输出：
      - MxN 的二值分配矩阵 assign_mat，assign_mat[i,j]=1 表示将检测 i 分配到对象 j。
        约束：每一行（检测）最多有一个 1；同一列（对象）可以被多个检测赋值。

    算法（当前仅支持 match_method=="sim_sum"）：
      - 将空间相似度与视觉相似度按 cfg.phys_bias 加权求和得到综合相似度 sims
      - 对每个检测取该行的最大相似度对象作为候选
      - 按检测的最大相似度从高到低遍历：只有当该最大相似度 > cfg.sim_threshold 时才做分配
        （贪心策略，优先确认高置信匹配）
    '''
    # 初始化分配矩阵为全零
    assign_mat = torch.zeros_like(spatial_sim)
    if cfg.match_method == "sim_sum":
        # 计算加权综合相似度，phys_bias控制物理空间相似度的权重
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim  # (M, N)
        # 计算每行的最大值及对应索引
        row_max, row_argmax = torch.max(sims, dim=1)  # (M,), (M,)
        # 从高到低处理，优先将高相似度的检测匹配到对象
        for i in row_max.argsort(descending=True):
            # 只有当相似度超过阈值时才进行匹配
            if row_max[i] > cfg.sim_threshold:
                assign_mat[i, row_argmax[i]] = 1
            else:
                # 由于是降序排序，后面的相似度只会更小，所以可以提前退出
                break
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")

    return assign_mat

def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float=0.025):
    '''
    为保存或可视化准备对象副本：
      - 深拷贝对象列表
      - 对每个对象的点云做体素下采样（减小体积、提升渲染速度）
      - 删除不必要的键，仅保留用于后续可视化/分析的字段

    输入：
      - objects: MapObjectList类型的对象列表
      - downsample_size: 体素下采样的体素大小，默认0.025米

    返回可序列化（JSON/pickle 友好）的对象结构。
    '''
    # 深拷贝对象列表，避免修改原始数据
    objects_to_save = copy.deepcopy(objects)

    # 对每个对象的点云进行体素下采样
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # 删除不必要的键，只保留需要用于可视化和分析的字段
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in [
                'pcd', 'bbox', 'clip_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color'
            ]:
                del objects_to_save[i][k]

    # 转换为可序列化的格式
    return objects_to_save.to_serializable()
    
def process_cfg(cfg: DictConfig):
    '''
    对 hydra/omegaconf 的 cfg 做小幅处理：
      - 将路径字符串转换为 Path
      - 若未在命令行或 cfg 中指定图像尺寸，则从数据集配置中读取（大多数数据集 RGB/Depth 分辨率相同）

    输入：
      - cfg: Hydra配置对象

    返回修改后的 cfg（原地修改）。
    '''
    # 将路径字符串转换为Path对象
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)

    # 对于大多数数据集，RGB和深度图具有相同的分辨率
    if cfg.dataset_config.name != "multiscan.yaml":
        # 从数据集配置中读取图像高度和宽度
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        # 对于multiscan数据集，RGB和深度图分辨率不同，必须显式指定图像尺寸
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg
    
@hydra.main(version_base=None, config_path="../configs/slam_pipeline", config_name="base")
def main(cfg : DictConfig):
    '''
    主函数，负责运行整个SLAM管道：
    1. 初始化数据集和配置
    2. 逐帧处理RGB、深度、位姿和GSA检测结果
    3. 将检测结果融合到三维地图中
    4. 执行周期性后处理（去噪、过滤、合并）
    5. 保存结果和可视化
    
    输入：
      - cfg: Hydra配置对象，包含所有运行参数
    '''
    # 处理配置对象
    cfg = process_cfg(cfg)
    
    # 初始化数据集，加载RGB、深度、位姿等数据
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
     
    # 设置类别颜色文件路径
    # cfg.color_file_name = '/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/gsa_classes_ram_mobilesam_withbg_allclasses'
    
    # 创建或加载类别和对应的颜色
    # classes, class_colors = create_or_load_colors(cfg, cfg.color_file_name)
    
    # Modified to load classes from global_class_counts.json
    dataset_output_path = Path("/home/zhengwu/Desktop/3D-Gaussian/GaussianGraph/dataset")
    global_counts_path = dataset_output_path / "global_class_counts.json"
    
    if os.path.exists(global_counts_path):
        import json
        with open(global_counts_path, 'r') as f:
            global_class_counts = json.load(f)
        classes = list(global_class_counts.keys())
        
        # Load or generate colors
        colors_path = dataset_output_path / "class_colors.json"
        if os.path.exists(colors_path):
            with open(colors_path, 'r') as f:
                class_colors = json.load(f)
            print(f"Loaded class colors from {colors_path}")
        else:
            from submodules.slam.utils import get_classes_colors
            class_colors = get_classes_colors(classes)
            # Convert keys to string for JSON serialization
            class_colors = {str(k): v for k, v in class_colors.items()}
            with open(colors_path, 'w') as f:
                json.dump(class_colors, f)
            print(f"Generated and saved class colors to {colors_path}")
    else:
        print(f"Warning: {global_counts_path} not found. Using empty classes.")
        classes = []
        class_colors = {}
    
    # 初始化地图对象列表
    objects = MapObjectList(device=cfg.device)
    
    # 初始化背景对象字典，如果不禁用背景处理
    if not cfg.skip_bg:
        # 背景类别单独处理，每个类别在地图中被表示为单个对象
        bg_objects = {
            c: None for c in BG_CLASSES
        }
    else:
        bg_objects = None
         
    # 初始化可视化渲染器（如果需要）
    if cfg.vis_render:
        # 加载预定义的相机视角参数
        view_param = o3d.io.read_pinhole_camera_parameters(cfg.render_camera_path)
            
        # 创建在线对象渲染器
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []
        
    # 创建保存所有帧对象的文件夹（如果需要）
    if cfg.save_objects_all_frames:
        save_all_folder = cfg.dataset_root \
            / cfg.scene_id / "objects_all_frames" / f"{cfg.gsa_variant}_{cfg.save_suffix}"
        os.makedirs(save_all_folder, exist_ok=True)
    
    # 设置检测结果文件夹路径
    # cfg.detection_folder_name = '/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/gsa_detections_ram_mobilesam_withbg_allclasses'
    
    # 遍历数据集中的每帧
    for idx in trange(len(dataset)):
        # 获取彩色图像路径并打开
        color_path = dataset.color_paths[idx]
        image_original_pil = Image.open(color_path)
        
        # 从数据集中获取彩色图像张量、深度张量、内参等信息
        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]
        
        # 将彩色张量转换为numpy数组并调整到0-255范围
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"
        
        # 处理深度图，去除通道维度
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()

        # 提取相机内参矩阵
        cam_K = intrinsics.cpu().numpy()[:3, :3]
        
        # 初始化GSA检测结果变量
        gobs = None # stands for grounded SAM observations
        
        # 构建检测结果文件路径
        color_path = Path(color_path)
        # detections_path = cfg.dataset_root / cfg.scene_id / cfg.detection_folder_name / color_path.name
        # detections_path = detections_path.with_suffix(".pkl.gz")
        
        # Modified to use the new dataset path from 2D_info_extract.py
        dataset_output_path = Path("/home/zhengwu/Desktop/3D-Gaussian/GaussianGraph/dataset")
        detections_path = dataset_output_path / (color_path.stem + ".pkl.gz")
        
        color_path = str(color_path)
        detections_path = str(detections_path)
        
        # 加载压缩的检测结果
        if os.path.exists(detections_path):
            with gzip.open(detections_path, "rb") as f:
                gobs = pickle.load(f)
        else:
            print(f"Warning: Detection file not found at {detections_path}. Skipping frame.")
            continue

        # 获取未转换的相机位姿
        unt_pose = dataset.poses[idx]
        unt_pose = unt_pose.cpu().numpy()
        
        # 不应用任何额外变换
        adjusted_pose = unt_pose
            
        # 将GSA检测结果转换为检测列表（前景和背景分开）
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = cfg,
            image = image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = adjusted_pose,
            class_names = classes,
            BG_CLASSES = BG_CLASSES,
            color_path = color_path,
        )
        
        # 处理背景检测结果
        if len(bg_detection_list) > 0:
            for detected_object in bg_detection_list:
                class_name = detected_object['class_name'][0]
                if bg_objects[class_name] is None:
                    # 如果是该背景类的首次检测，直接添加到字典
                    bg_objects[class_name] = detected_object
                else:
                    # 否则将新检测与现有背景对象合并
                    matched_obj = bg_objects[class_name]
                    matched_det = detected_object
                    bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            
        # 如果没有前景检测，则跳过后续处理
        if len(fg_detection_list) == 0:
            continue
            
        # 计算包含数量（如果启用）
        if cfg.use_contain_number:
            # 获取所有检测的边界框
            xyxy = fg_detection_list.get_stacked_values_torch('xyxy', 0)
            # 计算每个边界框包含的区域数量
            contain_numbers = compute_2d_box_contained_batch(xyxy, cfg.contain_area_thresh)
            # 将包含数量添加到每个检测中
            for i in range(len(fg_detection_list)):
                fg_detection_list[i]['contain_number'] = [contain_numbers[i]]
            
        # 如果地图中还没有对象，直接添加所有检测
        if len(objects) == 0:
            # 将所有检测添加到地图
            for i in range(len(fg_detection_list)):
                objects.append(fg_detection_list[i])

            # 跳过相似度计算，处理下一帧
            continue
                
        # 计算空间相似度和视觉相似度
        spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
        visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
        # 聚合相似度得到综合相似度
        agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)
        
        # 如果启用包含数量，则根据包含状态调整相似度
        if cfg.use_contain_number:
            # 获取所有对象的包含数量
            contain_numbers_objects = torch.Tensor([obj['contain_number'][0] for obj in objects])
            # 确定哪些检测和对象包含其他物体
            detection_contained = contain_numbers > 0 # (M,)
            object_contained = contain_numbers_objects > 0 # (N,)
            # 扩展维度以进行矩阵操作
            detection_contained = detection_contained.unsqueeze(1) # (M, 1)
            object_contained = object_contained.unsqueeze(0) # (1, N)                

            # 计算包含状态不匹配的位置，并对这些位置的相似度进行惩罚
            xor = detection_contained ^ object_contained
            agg_sim[xor] = agg_sim[xor] - cfg.contain_mismatch_penalty
        
        # 对相似度进行阈值处理，低于阈值的值设为负无穷
        agg_sim[agg_sim < cfg.sim_threshold] = float('-inf')
        
        # 根据相似度将检测结果合并到地图对象中
        objects = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim)
        
        # 周期性执行后处理（如果配置了的话）
        if cfg.denoise_interval > 0 and (idx+1) % cfg.denoise_interval == 0:
            objects = denoise_objects(cfg, objects)
        if cfg.filter_interval > 0 and (idx+1) % cfg.filter_interval == 0:
            objects = filter_objects(cfg, objects)
        if cfg.merge_interval > 0 and (idx+1) % cfg.merge_interval == 0:
            objects = merge_objects(cfg, objects)
            
        # 保存所有帧的对象（如果启用）
        if cfg.save_objects_all_frames:
            save_all_path = save_all_folder / f"{idx:06d}.pkl.gz"
            # 只保存检测次数大于等于最小阈值的对象
            objects_to_save = MapObjectList([
                _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            ])
            
            # 准备对象用于保存/可视化
            objects_to_save = prepare_objects_save_vis(objects_to_save)
            
            # 处理背景对象（如果需要）
            if not cfg.skip_bg:
                bg_objects_to_save = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
                bg_objects_to_save = prepare_objects_save_vis(bg_objects_to_save)
            else:
                bg_objects_to_save = None
            
            # 构建结果字典并保存
            result = {
                "camera_pose": adjusted_pose,
                "objects": objects_to_save,
                "bg_objects": bg_objects_to_save,
            }
            with gzip.open(save_all_path, 'wb') as f:
                pickle.dump(result, f)
        
        # 渲染可视化（如果启用）
        if cfg.vis_render:
            # 准备用于可视化的对象列表（深拷贝避免修改原始数据）
            objects_vis = MapObjectList([
                copy.deepcopy(_) for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            ])
            
            # 根据配置进行着色
            if cfg.class_agnostic:
                # 按实例着色（每个对象一个颜色）
                objects_vis.color_by_instance()
            else:
                # 按最常见类别着色
                objects_vis.color_by_most_common_classes(class_colors)
            
            # 渲染当前帧
            rendered_image, vis = obj_renderer.step(
                image = image_original_pil,
                gt_pose = adjusted_pose,
                new_objects = objects_vis,
                paint_new_objects=False,
                return_vis_handle = cfg.debug_render,
            )

            # 如果启用调试渲染，则显示可视化窗口
            if cfg.debug_render:
                vis.run()
                del vis
            
            # 将渲染图像转换为uint8并添加到帧列表
            if rendered_image is not None:
                rendered_image = (rendered_image * 255).astype(np.uint8)
                frames.append(rendered_image)
    
    # 所有帧处理完成后，对背景对象进行后处理
    if bg_objects is not None:
        bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
        bg_objects = denoise_objects(cfg, bg_objects)
        
    # 对前景对象进行去噪处理
    objects = denoise_objects(cfg, objects)
    
    # 保存完整点云（后处理前）
    if cfg.save_pcd:
        # 生成时间戳用于文件名
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建结果字典
        results = {
            'objects': objects.to_serializable(),
            'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
            'cfg': cfg,
            'class_names': classes,
            'class_colors': class_colors,
        }

        # 创建保存路径并确保目录存在
        pcd_save_path = cfg.dataset_root / \
            cfg.scene_id / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}.pkl.gz"
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
        pcd_save_path = str(pcd_save_path)
        
        # 保存结果
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud to {pcd_save_path}")
    
    # 执行最终的后处理（过滤和合并）
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    
    # 保存后处理后的完整点云
    if cfg.save_pcd:
        results['objects'] = objects.to_serializable()
        pcd_save_path = pcd_save_path[:-7] + "_post.pkl.gz"
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud after post-processing to {pcd_save_path}")
        
    # 保存元数据（如果启用了保存所有帧对象）
    if cfg.save_objects_all_frames:
        save_meta_path = save_all_folder / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': classes,
                'class_colors': class_colors,
            }, f)
        
    # 保存可视化视频（如果启用）
    if cfg.vis_render:
        # 渲染后处理后的最终帧
        objects_vis = MapObjectList([
            _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
        ])

        if cfg.class_agnostic:
            objects_vis.color_by_instance()
        else:
            objects_vis.color_by_most_common_classes(class_colors)
        
        rendered_image, vis = obj_renderer.step(
            image = image_original_pil,
            gt_pose = adjusted_pose,
            new_objects = objects_vis,
            paint_new_objects=False,
            return_vis_handle = False,
        )
        
        # 转换为uint8并添加到帧列表
        rendered_image = (rendered_image * 255).astype(np.uint8)
        frames.append(rendered_image)
        
        # 将帧列表保存为mp4视频
        frames = np.stack(frames)
        video_save_path = (
            cfg.dataset_root
            / cfg.scene_id
            / ("objects_mapping-%s-%s.mp4" % (cfg.gsa_variant, cfg.save_suffix))
        )
        imageio.mimwrite(video_save_path, frames, fps=10)
        print("Save video to %s" % video_save_path)
        
if __name__ == "__main__":
    main()