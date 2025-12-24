import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from collections import defaultdict, Counter

# 将当前工作目录添加到系统路径中，确保可以正确导入项目中的模块
sys.path.append(os.getcwd())

# 导入项目自定义模块
from scene import Scene, GaussianModel  # 场景和高斯模型类
from gaussian_renderer import render    # 渲染函数
from utils.opengs_utlis import load_code_book, calculate_iou  # 工具函数：加载码本，计算IoU
from arguments import ModelParams, PipelineParams, OptimizationParams  # 参数类

def get_mask_from_id(seg_maps, mask_id_dict):
    """
    根据 mask_id 字典和分割图 (seg_maps) 重建 2D 物体的二进制掩码。
    
    参数:
        seg_maps: [4, H, W] 的 numpy 数组或 tensor，包含不同层级的分割信息 (例如 SAM 的不同输出层级)
        mask_id_dict: 字典 {'default': id, 's': id, ...}，包含不同层级对应的 ID
    
    返回:
        mask: 二进制掩码 Tensor (布尔类型)
    """
    # 逻辑说明：为了简单起见，优先使用 'default' 层级，因为它通常包含最具代表性的掩码。
    
    # 如果输入是 numpy 数组，转换为 PyTorch tensor
    if isinstance(seg_maps, np.ndarray):
        seg_maps = torch.from_numpy(seg_maps)
    
    # 定义层级名称到索引的映射关系
    # 0:default (默认), 1:s (small, 小), 2:m (medium, 中), 3:l (large, 大)
    level_map = {'default': 0, 's': 1, 'm': 2, 'l': 3}
    
    # 首先尝试使用 'default' 层级
    if 'default' in mask_id_dict:
        level_idx = level_map['default']  # 获取层级索引
        mask_idx = mask_id_dict['default'] # 获取该层级下的 mask ID
        mask = (seg_maps[level_idx] == mask_idx) # 生成二进制掩码
        return mask
    
    # 如果没有 default 层级，回退到其他可用的层级
    for level, idx in mask_id_dict.items():
        if level in level_map:
            level_idx = level_map[level]
            mask = (seg_maps[level_idx] == idx)
            return mask
            
    # 如果没有找到匹配的层级，返回 None
    return None

def post_process_graph(graph):
    """
    对生成的场景图进行后处理，优化节点和边。
    
    主要步骤：
    1. 过滤掉类别为 'unknown' 的无效节点。
    2. 根据几何特征（距离、IoU）合并重复或过度分割的节点。
    3. 更新边关系以匹配合并后的节点。
    """
    # ==========================================
    # 1. 过滤未知类别 (Filter unknowns)
    # ==========================================
    valid_ids = set() # 存储有效节点的 ID
    new_nodes = []    # 存储过滤后的节点列表
    
    # 遍历所有节点，保留非 'unknown' 的节点
    for node in graph['nodes']:
        if node['category'] != 'unknown':
            new_nodes.append(node)
            valid_ids.add(node['id'])
    
    # 更新图中的节点列表
    graph['nodes'] = new_nodes
    
    # 过滤边，只保留连接两个有效节点的边
    new_edges = []
    for edge in graph['edges']:
        if edge['source'] in valid_ids and edge['target'] in valid_ids:
            new_edges.append(edge)
    graph['edges'] = new_edges

    # ==========================================
    # 2. 合并特定类别 (Merge specific categories)
    # ==========================================
    # 合并策略: 
    # 1. 激进合并 (Aggressive merging): 针对背景/大结构 (如墙, 地板等)，基于中心点距离进行合并。
    # 2. 保守合并 (Conservative merging): 针对所有类别，基于 3D 包围盒重叠率 (IoU) 合并。
    #    如果两个同类物体在 3D 空间中有显著重叠，它们很可能是同一个物体被过度分割了。
    
    # 定义需要激进合并的类别列表
    aggressive_merge_cats = ['Wall', 'Floor', 'Ceiling', 'Rug', 'Curtain', 'Blinds']
    
    id_mapping = {} # 记录旧 ID 到新 ID 的映射关系，用于后续更新边
    
    # --- 辅助函数定义开始 ---
    
    # 计算两个节点中心点的欧氏距离
    def get_dist(n1, n2):
        c1 = np.array(n1['center'])
        c2 = np.array(n2['center'])
        return np.linalg.norm(c1 - c2)

    # 计算两个节点 3D 包围盒的交并比 (IoU)
    def get_bbox_iou(n1, n2):
        b1 = np.array(n1['bbox']) # 格式: [[minx, miny, minz], [maxx, maxy, maxz]]
        b2 = np.array(n2['bbox'])
        
        # 计算交集区域 (Intersection) 的最小和最大坐标
        min_inter = np.maximum(b1[0], b2[0])
        max_inter = np.minimum(b1[1], b2[1])
        
        # 计算交集尺寸，如果有负值说明无交集，置为 0
        dims_inter = np.maximum(0, max_inter - min_inter)
        vol_inter = np.prod(dims_inter) # 计算交集体积
        
        if vol_inter == 0:
            return 0.0
            
        # 计算各自的体积
        dims1 = np.maximum(0, b1[1] - b1[0])
        dims2 = np.maximum(0, b2[1] - b2[0])
        vol1 = np.prod(dims1)
        vol2 = np.prod(dims2)
        
        # 计算并集体积 (Union)
        vol_union = vol1 + vol2 - vol_inter
        return vol_inter / (vol_union + 1e-6) # 返回 IoU，加小量防止除零

    # 检查是否存在包含关系或显著重叠
    def check_overlap(n1, n2):
        b1 = np.array(n1['bbox'])
        b2 = np.array(n2['bbox'])
        
        # 计算交集
        min_inter = np.maximum(b1[0], b2[0])
        max_inter = np.minimum(b1[1], b2[1])
        dims_inter = np.maximum(0, max_inter - min_inter)
        vol_inter = np.prod(dims_inter)
        
        if vol_inter == 0:
            return False
            
        # 计算各自体积
        dims1 = np.maximum(0, b1[1] - b1[0])
        dims2 = np.maximum(0, b2[1] - b2[0])
        vol1 = np.prod(dims1)
        vol2 = np.prod(dims2)
        
        # 如果交集覆盖了较小物体体积的 20% 以上，则认为需要合并
        min_vol = min(vol1, vol2)
        if min_vol == 0: return False
        
        return (vol_inter / min_vol) > 0.2

    # --- 辅助函数定义结束 ---

    # 按类别对节点进行分组，方便同类合并
    nodes_by_cat = defaultdict(list)
    for node in graph['nodes']:
        nodes_by_cat[node['category']].append(node)
        
    final_nodes = [] # 存储最终合并后的节点
    
    # 对每个类别内的节点进行遍历和合并处理
    for cat, nodes in nodes_by_cat.items():
        # 按点数 (point_count) 降序排序，保留点数最多的作为基准 (Base)
        nodes.sort(key=lambda x: x['point_count'], reverse=True)
        
        merged_nodes = []
        
        # 贪心合并过程
        while nodes:
            base = nodes.pop(0) # 取出当前最大的节点作为基准
            merged_nodes.append(base)
            
            to_merge = []   # 待合并的节点列表
            remaining = []  # 不合并，留待下一轮处理的节点列表
            
            for other in nodes:
                should_merge = False
                
                # 规则 1: 对背景类别进行激进合并
                if cat in aggressive_merge_cats:
                    if get_dist(base, other) < 2.5: # 如果中心点距离小于 2.5，则合并
                        should_merge = True
                
                # 规则 2: 对所有类别进行几何重叠检查
                # 如果它们显著重叠，它们很可能是同一个物体
                if not should_merge:
                    if check_overlap(base, other):
                        should_merge = True
                
                if should_merge:
                    to_merge.append(other)
                else:
                    remaining.append(other)
            
            # 更新待处理列表
            nodes = remaining
            
            # 执行合并操作
            for other in to_merge:
                # 记录 ID 映射：将 other 的 ID 映射到 base 的 ID
                id_mapping[other['id']] = base['id']
                
                # 更新统计信息
                # 1. 中心点: 按点数加权平均
                w1 = base['point_count']
                w2 = other['point_count']
                c1 = np.array(base['center'])
                c2 = np.array(other['center'])
                new_center = (c1 * w1 + c2 * w2) / (w1 + w2)
                base['center'] = new_center.tolist()
                
                # 2. 包围盒: 取两者的并集 (最小的 min 和最大的 max)
                b1 = np.array(base['bbox'])
                b2 = np.array(other['bbox'])
                new_min = np.minimum(b1[0], b2[0])
                new_max = np.maximum(b1[1], b2[1])
                base['bbox'] = [new_min.tolist(), new_max.tolist()]
                
                # 3. 点数: 累加
                base['point_count'] += other['point_count']
                
                # 4. 描述 (Caption): 保留最详细（通常是最长）的那个
                if len(other['caption']) > len(base['caption']):
                    base['caption'] = other['caption']
        
        final_nodes.extend(merged_nodes)
                
    graph['nodes'] = final_nodes
    
    # ==========================================
    # 3. 更新边信息 (Update Edges)
    # ==========================================
    # 使用 id_mapping 更新边的源和目标 ID
    updated_edges = []
    
    for edge in graph['edges']:
        s = edge['source']
        t = edge['target']
        
        # 如果 ID 在映射中，则替换为新的 ID；否则保持原样
        s = id_mapping.get(s, s)
        t = id_mapping.get(t, t)
        
        # 移除自环 (源和目标相同)
        if s == t:
            continue
            
        edge['source'] = s
        edge['target'] = t
        updated_edges.append(edge)
        
    # 去重边: 如果多条边连接相同的 (s, t)，则合并它们的关系列表
    final_edges_map = defaultdict(list)
    for edge in updated_edges:
        final_edges_map[(edge['source'], edge['target'])].extend(edge['relations'])
        
    final_edges = []
    for (s, t), rels in final_edges_map.items():
        # 再次投票选择最佳关系描述
        if rels:
            # 选择最详细（最长）的关系描述
            longest_rel = max(rels, key=len)
            final_rels = [longest_rel]
        else:
            final_rels = []
            
        final_edges.append({
            'source': s,
            'target': t,
            'relations': final_rels
        })
        
    graph['edges'] = final_edges
    
    # 返回处理后的图和 id_mapping，以便调用者如果需要可以重新计算统计信息
    return graph, id_mapping

def build_scene_graph(dataset, opt, pipe, iteration):
    """
    构建场景图的主函数。
    流程:
    1. 加载 Gaussian 模型和聚类信息 (Leaf Codebook)。
    2. 计算每个聚类的 3D 属性 (中心, 包围盒, 点数)。
    3. 遍历训练视角，渲染聚类掩码，结合 2D 语义信息 (JSON) 和 3D 投影建立节点和边。
    4. 聚合信息并进行后处理 (合并、过滤)。
    5. 保存最终的场景图 JSON。
    """
    with torch.no_grad(): # 禁用梯度计算，节省显存
        # ==========================================
        # 1. 加载 Gaussian 模型和场景
        # ==========================================
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # ==========================================
        # 2. 加载聚类标签 (Leaf Indices)
        # ==========================================
        model_path = dataset.model_path
        # 构建 leaf_code_book 的路径
        leaf_code_book_path = os.path.join(model_path, 'point_cloud', f"iteration_{iteration}", "leaf_code_book")
        
        if not os.path.exists(leaf_code_book_path):
            print(f"Error: Leaf codebook not found at {leaf_code_book_path}")
            return

        print(f"Loading codebook from {leaf_code_book_path}")
        # 加载聚类中心和每个点的聚类索引
        leaf_center, leaf_indices = load_code_book(leaf_code_book_path)
        leaf_cluster_indices = torch.from_numpy(leaf_indices).cuda()
        
        num_clusters = leaf_cluster_indices.max().item() + 1
        print(f"Number of clusters: {num_clusters}")

        # ==========================================
        # 2.1 计算 3D 聚类统计信息 (中心, 包围盒)
        # ==========================================
        xyz = gaussians.get_xyz.detach() # 获取所有高斯点的坐标 [N, 3]
        cluster_stats = {}
        unique_clusters = torch.unique(leaf_cluster_indices) # 获取所有唯一的聚类 ID
        
        print("Calculating 3D cluster statistics...")
        for cid in tqdm(unique_clusters, desc="Cluster Stats"):
            cid = cid.item()
            # 获取属于当前聚类的点的掩码
            mask = (leaf_cluster_indices == cid)
            points = xyz[mask]
            
            if points.shape[0] == 0:
                continue
                
            # 使用 float64 进行计算以保持精度
            points_d = points.double()
            
            # 使用均值作为中心 (接近几何质心/期望)
            center = points_d.mean(dim=0).cpu().tolist()
            
            # 使用分位数 (5% - 95%) 计算包围盒，以过滤掉漂浮点/离群点
            # 这可以防止少量杂散点导致包围盒过大
            min_pt = torch.quantile(points_d, 0.05, dim=0).cpu().tolist()
            max_pt = torch.quantile(points_d, 0.95, dim=0).cpu().tolist()
            
            # 四舍五入以确保 JSON 输出整洁但保持高精度 (例如 6 位小数)
            center = [round(x, 6) for x in center]
            min_pt = [round(x, 6) for x in min_pt]
            max_pt = [round(x, 6) for x in max_pt]
            
            cluster_stats[cid] = {
                'center': center,
                'bbox': [min_pt, max_pt], # 格式: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
                'point_count': points.shape[0]
            }

        # ==========================================
        # 3. 初始化图数据结构
        # ==========================================
        # 节点: 字典，key 为 cluster_id，value 包含类别列表和描述列表
        nodes = defaultdict(lambda: {'categories': [], 'captions': []})
        # 边: 字典，key 为 (id1, id2) 元组，value 为关系列表
        edges = defaultdict(list)

        # 设置背景颜色 (用于渲染)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda:0")

        # ==========================================
        # 4. 遍历训练相机 (Train Cameras)
        # ==========================================
        train_cameras = scene.getTrainCameras()
        
        for view in tqdm(train_cameras, desc="Processing Views"):
            # 确保数据在 GPU 上
            if not view.data_on_gpu:
                view.to_gpu()
                
            # 加载 2D 信息 (JSON 和 SegMaps)
            # dataset readers 将 _s.npy 加载到 view.sam_mask 中 (如果存在)
            # 但我们需要 _r.json 文件，它包含了语言特征和关系信息。
            
            # 构建 _r.json 的路径
            base_name = os.path.splitext(view.image_name)[0]
            json_path = os.path.join(dataset.source_path, "language_features", base_name + "_r.json")
            
            if not os.path.exists(json_path):
                # print(f"Warning: JSON not found for {view.image_name}")
                continue
                
            with open(json_path, 'r') as f:
                graph_data = json.load(f)
            
            # 检查是否有分割图 (SegMaps)
            if view.original_sam_mask is None:
                continue
            
            seg_maps = view.original_sam_mask.cuda() # 形状: [4, H, W]
            
            # 渲染当前视角的聚类
            # 我们使用带有 leaf_cluster_idx 的 render 函数
            # render_cluster=True 表示我们需要渲染聚类索引图
            
            render_pkg = render(view, gaussians, pipe, background, iteration,
                                leaf_cluster_idx=leaf_cluster_indices,
                                render_cluster=True,
                                better_vis=False, # better_vis=False 表示不要过于激进地过滤
                                rescale=False,
                                render_feat_map=False)
            
            rendered_masks = render_pkg["leaf_cluster_silhouettes"] # 形状: [N_visible, H, W]
            occured_ids = render_pkg["occured_leaf_id"] # 当前视角可见的聚类 ID 列表
            
            if rendered_masks is None or len(occured_ids) == 0:
                continue
                
            # 如果是列表则转换为 tensor
            if isinstance(rendered_masks, list):
                 rendered_masks = torch.stack(rendered_masks)
            
            # 确保为布尔值以进行位运算 (intersection/union)
            if rendered_masks.dtype == torch.float32 or rendered_masks.dtype == torch.float16:
                rendered_masks = rendered_masks > 0.5
            
            # ==========================================
            # 4.1 建立映射: 2D 物体 -> 3D 聚类
            # ==========================================
            # graph_data['captions'] 包含带有 'id' (mask_id) 的对象列表
            
            obj_to_cluster = {} # 2D_index -> 3D_cluster_id 的映射
            
            for i, obj in enumerate(graph_data['captions']):
                mask_id = obj['id']
                # 从分割图中提取当前物体的 2D 掩码
                obj_mask = get_mask_from_id(seg_maps, mask_id)
                
                if obj_mask is None:
                    continue
                
                # 计算该 2D 物体掩码与所有渲染出的 3D 聚类掩码的 IoU
                # obj_mask: [H, W]
                # rendered_masks: [N, H, W]
                # 利用广播机制一次性计算所有 IoU
                
                intersection = (rendered_masks & obj_mask).sum(dim=(1, 2)).float()
                union = (rendered_masks | obj_mask).sum(dim=(1, 2)).float()
                ious = intersection / (union + 1e-6)
                
                # 找到 IoU 最高的聚类作为匹配项
                best_iou, best_idx = ious.max(dim=0)
                
                if best_iou > 0.3: # IoU 阈值，大于 0.3 认为匹配成功
                    cluster_id = occured_ids[best_idx.item()]
                    obj_to_cluster[i] = cluster_id
                    
                    # 聚合节点信息 (类别和描述)
                    category = obj.get('class_name', '') # 获取类别名称
                    if not category:
                         # 如果不存在 class_name，这里可以添加备用逻辑
                         pass
                    
                    description = obj.get('description', '') # 获取描述
                    
                    nodes[cluster_id]['categories'].append(category)
                    nodes[cluster_id]['captions'].append(description)

            # ==========================================
            # 4.2 建立关系 (Edges)
            # ==========================================
            if 'relations' in graph_data:
                for rel in graph_data['relations']:
                    # rel 包含 'id_pair' (两个物体的 mask id)
                    
                    id_pair = rel.get('id_pair')
                    if not id_pair:
                        continue
                        
                    # id_pair 是 [mask_id_1, mask_id_2]
                    mask1 = get_mask_from_id(seg_maps, id_pair[0])
                    mask2 = get_mask_from_id(seg_maps, id_pair[1])
                    
                    if mask1 is None or mask2 is None:
                        continue
                        
                    # 内部辅助函数：为给定的 mask 找到最佳匹配的 3D 聚类
                    def find_best_cluster(mask, rendered_masks, occured_ids):
                        intersection = (rendered_masks & mask).sum(dim=(1, 2)).float()
                        union = (rendered_masks | mask).sum(dim=(1, 2)).float()
                        ious = intersection / (union + 1e-6)
                        best_iou, best_idx = ious.max(dim=0)
                        if best_iou > 0.1: # 关系匹配可以使用稍低的阈值
                            return occured_ids[best_idx.item()]
                        return None

                    # 找到关系双方对应的 3D 聚类 ID
                    c1 = find_best_cluster(mask1, rendered_masks, occured_ids)
                    c2 = find_best_cluster(mask2, rendered_masks, occured_ids)
                    
                    # 如果双方都找到了对应的聚类，且不是同一个聚类，则添加边
                    if c1 is not None and c2 is not None and c1 != c2:
                        text = rel.get('relationship', '') # 获取关系描述
                        edges[(c1, c2)].append(text)

            # 清理 GPU 显存
            if view.data_on_gpu:
                view.to_cpu()
            torch.cuda.empty_cache()

        # ==========================================
        # 5. 聚合和保存
        # ==========================================
        final_graph = {
            'nodes': [],
            'edges': []
        }
        
        print("Aggregating Graph...")
        
        # 处理节点信息
        for cid, info in nodes.items():
            # 使用多数投票机制确定最终类别
            cats = [c for c in info['categories'] if c]
            if cats:
                main_cat = Counter(cats).most_common(1)[0][0]
            else:
                main_cat = "unknown"
            
            # 根据多数类别过滤描述
            # 只保留与最终确定的类别一致的描述
            valid_captions = []
            for i, cat in enumerate(info['categories']):
                if cat == main_cat:
                    valid_captions.append(info['captions'][i])
            
            # 如果没有有效描述 (例如 main_cat 是 unknown)，回退到所有描述
            if not valid_captions:
                valid_captions = info['captions']

            # 选择唯一最准确和详细的描述
            # 策略: 从有效描述中选择最长的一个。长度通常与细节丰富程度相关。
            if valid_captions:
                full_caption = max(valid_captions, key=len)
            else:
                full_caption = ""
            
            # 获取之前计算的 3D 统计信息
            stats = cluster_stats.get(cid, {'center': [0,0,0], 'bbox': [[0,0,0],[0,0,0]], 'point_count': 0})

            final_graph['nodes'].append({
                'id': int(cid),
                'category': main_cat,
                'caption': full_caption,
                'center': stats['center'],
                'bbox': stats['bbox'],
                'point_count': stats['point_count']
            })
            
        # 处理边信息
        for (c1, c2), rels in edges.items():
            # 总结关系
            # 同样投票选择最详细（最长）的关系描述
            if rels:
                longest_rel = max(rels, key=len)
                # 保持列表格式以符合 schema 一致性
                final_rels = [longest_rel]
            else:
                final_rels = []

            final_graph['edges'].append({
                'source': int(c1),
                'target': int(c2),
                'relations': final_rels
            })
            
        output_path = os.path.join(model_path, f"scene_graph_{iteration}.json")
        
        # ==========================================
        # 6. 后处理 (Post-processing)
        # ==========================================
        # 过滤未知节点, 合并背景对象和重叠对象
        print("Post-processing Graph (Filtering & Merging)...")
        final_graph, id_mapping = post_process_graph(final_graph)

        # 重新计算合并后节点的中心/包围盒/点数
        # 因为 post_process_graph 只是合并了逻辑节点，我们需要更新物理属性
        
        # 构建反向映射: new_id -> list of original ids (合并了哪些原始聚类)
        rev_map = {}
        for orig_id in cluster_stats.keys():
            new_id = id_mapping.get(orig_id, orig_id)
            rev_map.setdefault(new_id, []).append(orig_id)

        # 使用重新计算的聚合统计信息更新 final_graph 节点
        new_nodes = []
        for node in final_graph['nodes']:
            nid = node['id']
            orig_ids = rev_map.get(nid, [nid])
            
            # 聚合计算
            total_count = 0
            sum_center = np.zeros(3)
            bbox_min = [float('inf')] * 3
            bbox_max = [float('-inf')] * 3
            
            for oid in orig_ids:
                stats = cluster_stats.get(oid)
                if stats is None:
                    continue
                cnt = stats['point_count']
                c = np.array(stats['center'], dtype=float)
                
                # 累加用于计算加权平均中心
                total_count += cnt
                sum_center += c * cnt
                
                # 更新包围盒范围
                bmin = np.array(stats['bbox'][0], dtype=float)
                bmax = np.array(stats['bbox'][1], dtype=float)
                bbox_min = np.minimum(bbox_min, bmin)
                bbox_max = np.maximum(bbox_max, bmax)
                
            if total_count > 0:
                agg_center = (sum_center / total_count).tolist()
            else:
                agg_center = node['center']
                
            # 更新节点属性
            node['center'] = [round(float(x), 6) for x in agg_center]
            node['bbox'] = [[float(x) for x in bbox_min], [float(x) for x in bbox_max]]
            node['point_count'] = int(total_count)
            new_nodes.append(node)

        final_graph['nodes'] = new_nodes

        # 保存最终结果
        with open(output_path, 'w') as f:
            json.dump(final_graph, f, indent=2)

        print(f"Scene Graph saved to {output_path}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = ArgumentParser(description="Build Scene Graph")
    
    # 添加各模块参数
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--iteration", type=int, default=30000, help="使用的迭代次数")
    
    args = parser.parse_args(sys.argv[1:])
    
    # 调用主函数
    build_scene_graph(lp.extract(args), op.extract(args), pp.extract(args), args.iteration)
