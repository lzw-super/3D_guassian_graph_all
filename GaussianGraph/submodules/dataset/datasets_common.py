"""
PyTorch dataset classes for datasets in the NICE-SLAM format.
Large chunks of code stolen and adapted from:
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py

Support for Replica (sequences from the iMAP paper), TUM RGB-D, NICE-SLAM Apartment.
TODO: Add Azure Kinect dataset support
"""

import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages

from conceptgraph.utils.general_utils import to_scalar, measure_time


def as_intrinsics_matrix(intrinsics):
    """
    将相机内参转换为矩阵表示。
    
    Args:
        intrinsics: 相机内参列表，格式为 [fx, fy, cx, cy]
        
    Returns:
        K: 相机内参矩阵，形状为 (3, 3)
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]  # 焦距 x
    K[1, 1] = intrinsics[1]  # 焦距 y
    K[0, 2] = intrinsics[2]  # 主点 x
    K[1, 2] = intrinsics[3]  # 主点 y
    return K

def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    '''
    从相机内参矩阵中提取 fx, fy, cx, cy 参数
    
    Args:
        K: 相机内参矩阵，形状为 (3, 3)
        
    Returns:
        tuple: (fx, fy, cx, cy) 四个标量值
    '''
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


def readEXR_onlydepth(filename):
    """
    从EXR图像文件中读取深度数据。
    
    Args:
        filename (str): 文件路径。
        
    Returns:
        Y (numpy.array): 深度缓冲区，float32格式。
    """
    # 将导入移至此处，因为只有CoFusion需要这些包
    # 有时安装openexr比较困难，如果没有openexr，仍然可以运行其他数据集
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


class GradSLAMDataset(torch.utils.data.Dataset):
    """
    GradSLAM数据集的基类，提供了处理RGB-D数据的通用功能
    
    该类是一个抽象基类，定义了SLAM系统中处理RGB-D数据所需的基本接口和功能。
    它继承自PyTorch的Dataset类，支持批量加载RGB-D图像、相机位姿和其他相关数据。
    具体的数据集实现（如ReplicaDataset）需要继承此类并实现特定的方法。
    """
    
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True, # 如果为True，位姿是相对于第一帧的
        **kwargs,
    ):
        """
        初始化GradSLAMDataset
        
        Args:
            config_dict: 包含数据集配置的字典，特别是相机参数
            stride: 帧采样间隔，默认为1（使用所有帧）
            start: 起始帧索引
            end: 结束帧索引，-1表示使用所有帧直到末尾
            desired_height: 输出图像的目标高度
            desired_width: 输出图像的目标宽度
            channels_first: 是否使用通道在前的格式 (C, H, W)
            normalize_color: 是否对颜色图像进行归一化
            device: 数据加载到的设备（CPU或GPU）
            dtype: 数据类型
            load_embeddings: 是否加载特征嵌入
            embedding_dir: 嵌入特征的目录名
            embedding_dim: 嵌入特征的维度
            relative_pose: 是否将位姿转换为相对于第一帧的形式
            **kwargs: 其他可选参数
        """
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        # 原始相机参数
        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        # 目标图像尺寸和下采样比例
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        # 嵌入特征相关设置
        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        # 帧范围设置和验证
        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )

        # 相机畸变参数（可选）
        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        # 裁剪大小（可选）
        self.crop_size = (
            config_dict["camera_params"]["crop_size"]
            if "crop_size" in config_dict["camera_params"]
            else None
        )

        # 边缘裁剪参数（可选）
        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        # 获取文件路径（由子类实现）
        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        if self.load_embeddings:
            if len(self.color_paths) != len(self.embedding_paths):
                raise ValueError(
                    "Mismatch between number of color images and number of embedding files."
                )
        self.num_imgs = len(self.color_paths)
        # 加载位姿（由子类实现）
        self.poses = self.load_poses()
        
        # 设置结束帧索引
        if self.end == -1:
            self.end = self.num_imgs

        # 应用帧采样（根据stride参数）
        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # 保留的索引张量
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # 更新图像数量
        self.num_imgs = len(self.color_paths)

        # 将位姿转换为张量
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            # 如果需要，将位姿转换为相对于第一帧的形式
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        """
        返回数据集中的样本数量
        
        Returns:
            int: 数据集中的图像/帧数量
        """
        return self.num_imgs

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            if self.embedding_dir == "embed_semseg":
                # embed_semseg is stored as uint16 pngs
                embedding_paths = natsorted(
                    glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.png")
                )
            else:
                embedding_paths = natsorted(
                    glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
                )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        """加载相机位姿。由子类实现。"""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray):
        r"""预处理彩色图像：调整大小为 :math:`(H, W, C)`，（可选）将值归一化到
        :math:`[0, 1]`，并（可选）使用通道优先的 :math:`(C, H, W)` 表示。

        Args:
            color (np.ndarray): 原始RGB图像输入

        Returns:
            np.ndarray: 预处理后的RGB图像

        Shape:
            - 输入: :math:`(H_\text{old}, W_\text{old}, C)`
            - 输出: 如果 `self.channels_first == False` 则为 :math:`(H, W, C)`，
                    否则为 :math:`(C, H, W)`。
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""预处理深度图像：调整大小，添加通道维度，并将值缩放为米。
        可选地将深度从通道在后的 :math:`(H, W, 1)` 转换为通道在前的 :math:`(1, H, W)` 表示。

        Args:
            depth (np.ndarray): 原始深度图像

        Returns:
            np.ndarray: 预处理后的深度图像

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - 输出: 如果 `self.channels_first == False` 则为 :math:`(H, W, 1)`，
                    否则为 :math:`(1, H, W)`。
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,  # 深度图使用最近邻插值避免伪影
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        # 将深度值从像素单位转换为米（除以缩放因子）
        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""预处理位姿：将序列中的第一个位姿设为单位矩阵，并计算所有其他位姿
        相对于第一个位姿的齐次变换矩阵。

        Args:
            poses (torch.Tensor): 待预处理的位姿矩阵

        Returns:
            Output (torch.Tensor): 预处理后的位姿

        Shape:
            - poses: :math:`(L, 4, 4)` 其中 :math:`L` 表示序列长度。
            - Output: :math:`(L, 4, 4)` 其中 :math:`L` 表示序列长度。
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        返回相机内参矩阵 K
        
        Returns:
            K (torch.Tensor): 相机内参矩阵，形状为 (3, 3)
        '''
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K
    
    def read_embedding_from_file(self, embedding_path: str):
        '''
        从文件中读取嵌入特征并进行处理。由子类为每个数据集单独实现。
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        """
        获取指定索引的样本数据
        
        Args:
            index: 样本索引
            
        Returns:
            tuple: 包含以下元素的元组
                - 预处理后的彩色图像（张量）
                - 预处理后的深度图像（张量）
                - 相机内参（张量）
                - 相机位姿（张量）
                - （可选）嵌入特征（张量，如果load_embeddings为True）
        """
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        # 读取彩色图像
        color = np.asarray(imageio.imread(color_path), dtype=float)
        # 预处理彩色图像
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        
        # 根据文件扩展名读取不同格式的深度图
        if ".png" in depth_path:
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        elif ".npy" in depth_path:
            depth = np.load(depth_path)
        else:
            raise NotImplementedError

        # 构建相机内参矩阵并转换为张量
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        
        # 如果有畸变参数，对彩色图像进行去畸变（注意：深度图不进行去畸变）
        if self.distortion is not None:
            color = cv2.undistort(color, K, self.distortion)

        # 预处理深度图像
        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        # 根据图像缩放比例调整内参矩阵
        K = datautils.scale_intrinsics(
            K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        # 构建4x4的内参矩阵（前3x3为原始内参，其他元素为0，右下角为1）
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        # 获取转换后的位姿
        pose = self.transformed_poses[index]

        # 如果需要加载嵌入特征，返回包含嵌入特征的元组
        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # 允许嵌入特征使用不同的数据类型
            )

        # 返回基本数据（无嵌入特征）
        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
        )


class ICLDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict: Dict,
        basedir: Union[Path, str],
        sequence: Union[Path, str],
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[Union[Path, str]] = "embeddings",
        embedding_dim: Optional[int] = 512,
        embedding_file_extension: Optional[str] = "pt",
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # Attempt to find pose file (*.gt.sim)
        self.pose_path = glob.glob(os.path.join(self.input_folder, "*.gt.sim"))
        if self.pose_path == 0:
            raise ValueError("Need pose file ending in extension `*.gt.sim`")
        self.pose_path = self.pose_path[0]
        self.embedding_file_extension = embedding_file_extension
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(
                    f"{self.input_folder}/{self.embedding_dir}/*.{self.embedding_file_extension}"
                )
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []

        lines = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        _posearr = []
        for line in lines:
            line = line.strip().split()
            if len(line) == 0:
                continue
            _npvec = np.asarray(
                [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            )
            _posearr.append(_npvec)
        _posearr = np.stack(_posearr)

        for pose_line_idx in range(0, _posearr.shape[0], 3):
            _curpose = np.zeros((4, 4))
            _curpose[3, 3] = 3
            _curpose[0] = _posearr[pose_line_idx]
            _curpose[1] = _posearr[pose_line_idx + 1]
            _curpose[2] = _posearr[pose_line_idx + 2]
            poses.append(torch.from_numpy(_curpose).float())

        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class ReplicaDataset(GradSLAMDataset):
    """
    Replica数据集的具体实现类
    
    该类继承自GradSLAMDataset，专门用于处理Replica数据集格式的RGB-D数据。
    Replica数据集是由iMAP论文中使用的室内场景数据集，提供了高质量的RGB图像、深度图和相机位姿。
    该实现负责从特定格式的文件结构中加载数据，并处理帧间的相机位姿关系。
    """
    
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        """
        初始化Replica数据集
        
        Args:
            config_dict: 包含数据集配置的字典，特别是相机参数
            basedir: 数据集的基础目录路径
            sequence: 序列名称或子目录名称
            stride: 帧采样间隔，默认为None（相当于1）
            start: 起始帧索引
            end: 结束帧索引，-1表示使用所有帧直到末尾
            desired_height: 输出图像的目标高度
            desired_width: 输出图像的目标宽度
            load_embeddings: 是否加载特征嵌入
            embedding_dir: 嵌入特征的目录名
            embedding_dim: 嵌入特征的维度
            **kwargs: 传递给父类GradSLAMDataset的其他参数
        """
        # 设置输入文件夹路径，格式为 basedir/sequence
        self.input_folder = os.path.join(basedir, sequence)
        # 设置位姿文件路径，Replica数据集的位姿通常存储在traj.txt文件中
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        
        # 调用父类构造函数，传递所有相关参数
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        """
        获取彩色图像、深度图像和嵌入特征的文件路径
        
        该方法根据Replica数据集的文件结构，查找并返回所有需要的文件路径。
        Replica数据集的RGB图像通常存储在results子目录中，命名格式为frame*.jpg。
        深度图也存储在results子目录中，命名格式为depth*.png。
        
        Returns:
            tuple: (color_paths, depth_paths, embedding_paths) 三个列表，分别包含
                  彩色图像、深度图像和嵌入特征（如果启用）的文件路径
        """
        # 使用natsorted确保文件按数字顺序排序
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        embedding_paths = None
        
        # 如果需要加载嵌入特征，则查找嵌入特征文件
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        """
        从位姿文件加载相机位姿
        
        该方法读取Replica数据集的traj.txt文件，解析其中的相机位姿信息。
        每个位姿被解析为4x4的齐次变换矩阵，用于表示相机在世界坐标系中的位置和方向。
        
        Returns:
            list: 包含所有帧相机位姿的列表，每个元素是一个torch.Tensor类型的4x4变换矩阵
        """
        poses = []
        # 打开位姿文件
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        
        # 读取每一行对应的位姿
        for i in range(self.num_imgs):
            line = lines[i]
            # 将每一行数据解析为4x4的齐次变换矩阵
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # 注意：以下两行代码被注释掉了，可能是因为原始数据已经是正确的坐标系
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # 转换为PyTorch张量
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        """
        从文件中读取嵌入特征
        
        该方法从指定路径加载嵌入特征文件，这些文件通常是预计算的特征向量，
        用于增强SLAM系统的语义理解能力。
        
        Args:
            embedding_file_path: 嵌入特征文件的路径
            
        Returns:
            torch.Tensor: 处理后的嵌入特征张量，形状为(1, H, W, embedding_dim)
        """
        # 加载PyTorch格式的嵌入特征文件
        embedding = torch.load(embedding_file_path)
        # 调整张量维度顺序，从(1, embedding_dim, H, W)转换为(1, H, W, embedding_dim)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class ScannetDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None

        # Load the intrinsic matrix from the file in each scene
        scene_intrinsic_path = os.path.join(self.input_folder, "intrinsic", "intrinsic_color.txt")
        scene_intrinsic = np.loadtxt(scene_intrinsic_path)
        config_dict['camera_params']['fx'] = scene_intrinsic[0, 0]
        config_dict['camera_params']['fy'] = scene_intrinsic[1, 1]
        config_dict['camera_params']['cx'] = scene_intrinsic[0, 2]
        config_dict['camera_params']['cy'] = scene_intrinsic[1, 2]
        
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        print(embedding_file_path)
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class Ai2thorDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            if self.embedding_dir == "embed_semseg":
                # embed_semseg is stored as uint16 pngs
                embedding_paths = natsorted(
                    glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.png")
                )
            else:
                embedding_paths = natsorted(
                    glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
                )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        if self.embedding_dir == "embed_semseg":
            embedding = imageio.imread(embedding_file_path) # (H, W)
            embedding = cv2.resize(
                embedding, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST
            )
            embedding = torch.from_numpy(embedding).long() # (H, W)
            embedding = F.one_hot(embedding, num_classes = self.embedding_dim) # (H, W, C)
            embedding = embedding.half() # (H, W, C)
            embedding = embedding.permute(2, 0, 1) # (C, H, W)
            embedding = embedding.unsqueeze(0) # (1, C, H, W)
        else:
            embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

class AzureKinectDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None

        # check if a file named 'poses_global_dvo.txt' exists in the basedir / sequence folder 
        if os.path.isfile(os.path.join(basedir, sequence, 'poses_global_dvo.txt')):
            self.pose_path = os.path.join(basedir, sequence, 'poses_global_dvo.txt')
            
        # if "odomfile" in kwargs.keys():
        #     self.pose_path = kwargs["odomfile"]
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        if self.pose_path is None:
            print(
                "WARNING: Dataset does not contain poses. Returning identity transform."
            )
            return [torch.eye(4).float() for _ in range(self.num_imgs)]
        else:
            # Determine whether the posefile ends in ".log"
            # a .log file has the following format for each frame
            # frame_idx frame_idx+1
            # row 1 of 4x4 transform
            # row 2 of 4x4 transform
            # row 3 of 4x4 transform
            # row 4 of 4x4 transform
            # [repeat for all frames]
            #
            # on the other hand, the "poses_o3d.txt" or "poses_dvo.txt" files have the format
            # 16 entries of 4x4 transform
            # [repeat for all frames]
            if self.pose_path.endswith(".log"):
                # print("Loading poses from .log format")
                poses = []
                lines = None
                with open(self.pose_path, "r") as f:
                    lines = f.readlines()
                if len(lines) % 5 != 0:
                    raise ValueError(
                        "Incorrect file format for .log odom file "
                        "Number of non-empty lines must be a multiple of 5"
                    )
                num_lines = len(lines) // 5
                for i in range(0, num_lines):
                    _curpose = []
                    _curpose.append(list(map(float, lines[5 * i + 1].split())))
                    _curpose.append(list(map(float, lines[5 * i + 2].split())))
                    _curpose.append(list(map(float, lines[5 * i + 3].split())))
                    _curpose.append(list(map(float, lines[5 * i + 4].split())))
                    _curpose = np.array(_curpose).reshape(4, 4)
                    poses.append(torch.from_numpy(_curpose))
            else:
                poses = []
                lines = None
                with open(self.pose_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    if len(line.split()) == 0:
                        continue
                    c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                    poses.append(torch.from_numpy(c2w))
            return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding  # .permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class RealsenseDataset(GradSLAMDataset):
    """
    Dataset class to process depth images captured by realsense camera on the tabletop manipulator 
    """
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # only poses/images/depth corresponding to the realsense_camera_order are read/used
        self.pose_path = os.path.join(self.input_folder, "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg"))
        )
        depth_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png"))
        )
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        poses = []
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        for posefile in posefiles:
            c2w = torch.from_numpy(np.load(posefile)).float()
            _R = c2w[:3, :3]
            _t = c2w[:3, 3]
            _pose = P @ c2w @ P.T
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class Record3DDataset(GradSLAMDataset):
    """
    Dataset class to read in saved files from the structure created by our
    `save_record3d_stream.py` script
    """
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "poses")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, "rgb", "*.png"))
        )
        depth_paths = natsorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png"))
        )
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        posefiles = natsorted(glob.glob(os.path.join(self.pose_path, "*.npy")))
        poses = []
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        for posefile in posefiles:
            c2w = torch.from_numpy(np.load(posefile)).float()
            _R = c2w[:3, :3]
            _t = c2w[:3, 3]
            _pose = P @ c2w @ P.T
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class MultiscanDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, f"{sequence}.jsonl")
        
        scene_meta = json.load(
            open(os.path.join(self.input_folder, f"{sequence}.json"), "r")
        )
        cam_K = scene_meta['streams'][0]['intrinsics']
        cam_K = np.array(cam_K).reshape(3, 3).T
        
        config_dict['camera_params']['fx'] = cam_K[0, 0]
        config_dict['camera_params']['fy'] = cam_K[1, 1]
        config_dict['camera_params']['cx'] = cam_K[0, 2]
        config_dict['camera_params']['cy'] = cam_K[1, 2]
        config_dict["camera_params"]["image_height"] = scene_meta['streams'][0]['resolution'][0]
        config_dict["camera_params"]["image_width"] = scene_meta['streams'][0]['resolution'][1]
        
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/outputs/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/outputs/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
            
        return color_paths, depth_paths, embedding_paths
        
    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        n_sampled = len(glob.glob(f"{self.input_folder}/outputs/color/*.png"))
        step = round(len(lines) / float(n_sampled))

        poses = []
        for i in range(0, len(lines), step):
            line = lines[i]
            info = json.loads(line)
            transform = np.asarray(info.get('transform'))
            transform = np.reshape(transform, (4, 4), order='F')
            transform = np.dot(transform, np.diag([1, -1, -1, 1]))
            transform = transform / transform[3][3]
            poses.append(torch.from_numpy(transform).float())
            
        return poses
        
    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)


class Hm3dDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/*_depth.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/*.json"))
        
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        
        for posefile in posefiles:
            with open(posefile, 'r') as f:
                pose_raw = json.load(f)
            pose = np.asarray(pose_raw['pose'])
            
            pose = torch.from_numpy(pose).float()
            pose = P @ pose @ P.T
            
            poses.append(pose)
            
        return poses
    
class Hm3dOpeneqaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None

        # Load the intrinsic matrix from the file in each scene
        scene_intrinsic_path = os.path.join(self.input_folder, "intrinsic_color.txt")
        scene_intrinsic = np.loadtxt(scene_intrinsic_path)
        config_dict['camera_params']['fx'] = scene_intrinsic[0, 0]
        config_dict['camera_params']['fy'] = scene_intrinsic[1, 1]
        config_dict['camera_params']['cx'] = scene_intrinsic[0, 2]
        config_dict['camera_params']['cy'] = scene_intrinsic[1, 2]
        
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/*-rgb.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/*-depth.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/[0-9]*.txt"))

        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        
        for posefile in posefiles:
            pose = torch.from_numpy(np.loadtxt(posefile)).float()
            pose = P @ pose @ P.T
            poses.append(pose)
        return poses
        
def load_dataset_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def common_dataset_to_batch(dataset):
    colors, depths, poses = [], [], []
    intrinsics, embeddings = None, None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose, _embedding = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
        if _embedding is not None:
            if embeddings is None:
                embeddings = [_embedding]
            else:
                embeddings.append(_embedding)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    if embeddings is not None:
        embeddings = torch.stack(embeddings, dim=1)
        # # (1, NUM_IMG, DIM_EMBED, H, W) -> (1, NUM_IMG, H, W, DIM_EMBED)
        # embeddings = embeddings.permute(0, 1, 3, 4, 2)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()
    if embeddings is not None:
        embeddings = embeddings.float()
    return colors, depths, intrinsics, poses, embeddings

@measure_time
def get_dataset(dataconfig, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig)
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["multiscan"]:
        return MultiscanDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict['dataset_name'].lower() in ['hm3d']:
        return Hm3dDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict['dataset_name'].lower() in ['hm3d-openeqa']:
        return Hm3dOpeneqaDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


if __name__ == "__main__":
    cfg = load_dataset_config(
        "/home/qiao/src/gradslam-foundation/examples/dataconfigs/replica/replica.yaml"
    )
    dataset = ReplicaDataset(
        config_dict=cfg,
        basedir="/home/qiao/src/nice-slam/Datasets/Replica",
        sequence="office0",
        start=0,
        end=1900,
        stride=100,
        # desired_height=680,
        # desired_width=1200,
        desired_height=240,
        desired_width=320,
    )

    colors, depths, poses = [], [], []
    intrinsics = None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)
    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()

    # create rgbdimages object
    rgbdimages = RGBDImages(
        colors,
        depths,
        intrinsics,
        poses,
        channels_first=False,
        has_embeddings=False,  # KM
    )

    # SLAM
    slam = PointFusion(odom="gt", dsratio=1, device="cuda:0", use_embeddings=False)
    pointclouds, recovered_poses = slam(rgbdimages)

    import open3d as o3d

    print(pointclouds.colors_padded.shape)
    pcd = pointclouds.open3d(0)
    o3d.visualization.draw_geometries([pcd])

    # from icl_dataset import ICLWithCLIPEmbeddings

    # dataset = ICLWithCLIPEmbeddings(
    #     os.path.join("/home/krishna/data/icl/"),
    #     trajectories="living_room_traj1_frei_png",
    #     stride=10,
    #     height=480,
    #     width=640,
    #     embedding_dir="feat_lseg_240_320",
    # )
    # colors, depths, intrinsics, poses, _, _, embeddings = dataset[
    #     0
    # ]  # next(iter(loader))
    # print(colors.shape, depths.shape, intrinsics.shape, poses.shape)
