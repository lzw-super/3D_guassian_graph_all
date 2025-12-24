import torch
import numpy as np
import torch.nn.functional as F
import os
from bitarray import bitarray
from collections import OrderedDict

def calculate_pairwise_distances(tensor1, tensor2, metric=None):
    """
    Calculate L1 (Manhattan) and L2 (Euclidean) distances between every pair of vectors
    in two tensors of shape [m, 6] and [n, 6].
    Args:
        tensor1 (torch.Tensor): A tensor of shape [m, 6].
        tensor2 (torch.Tensor): Another tensor of shape [n, 6].
    Returns:
        torch.Tensor: L1 distances of shape [m, n].
        torch.Tensor: L2 distances of shape [m, n].
    """
    # Reshape tensors to allow broadcasting
    # tensor1 shape becomes [m, 1, 6] and tensor2 shape becomes [1, n, 6]
    tensor1 = tensor1.unsqueeze(1)  # Now tensor1 is [m, 1, 6]
    tensor2 = tensor2.unsqueeze(0)  # Now tensor2 is [1, n, 6]

    # Compute the L1 distance
    if metric == "l1":
        return torch.abs(tensor1 - tensor2).sum(dim=2), None  # Result is [m, n]

    # Compute the L2 distance
    if metric == "l2":
        return None, torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=2))  # Result is [m, n]

    l1_distances = torch.abs(tensor1 - tensor2).sum(dim=2)
    l2_distances = torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=2))
    return l1_distances, l2_distances

def calculate_distances(tensor1, tensor2, metric=None):
    """
    Calculate L1 (Manhattan) and L2 (Euclidean) distances between corresponding vectors
    in two tensors of shape [N, dim].
    Args:
        tensor1 (torch.Tensor): A tensor of shape [N, dim].
        tensor2 (torch.Tensor): Another tensor of shape [N, dim].
    Returns:
        torch.Tensor: L1 distances of shape [N].
        torch.Tensor: L2 distances of shape [N].
    """
    # Compute L1 distance
    if metric == "l1":
        return torch.abs(tensor1 - tensor2).sum(dim=1)
    
    # Compute L2 distance
    if metric == "l2":
        return torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=1))
    
    l1_distances = torch.abs(tensor1 - tensor2).sum(dim=1)
    l2_distances = torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=1))

    return l1_distances, l2_distances
    

def bin2dec(b, bits):
    """Convert binary b to decimal integer.
    Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, torch.int64)
    return torch.sum(mask * b, -1)

def load_code_book(base_path):
    inds_file = os.path.join(base_path, 'kmeans_inds.bin')
    codebook_file = os.path.join(base_path, 'kmeans_centers.pth')
    args_file = os.path.join(base_path, 'kmeans_args.npy')
    codebook = torch.load(codebook_file)    # [num_cluster, dim]
    args_dict = np.load(args_file, allow_pickle=True).item()
    quant_params = args_dict['params']
    loaded_bitarray = bitarray()
    with open(inds_file, 'rb') as file:
        loaded_bitarray.fromfile(file)
    # bitarray pads 0s if array is not divisible by 8. ignore extra 0s at end when loading
    total_len = args_dict['total_len']
    loaded_bitarray = loaded_bitarray[:total_len].tolist()
    indices = np.reshape(loaded_bitarray, (-1, args_dict['n_bits']))
    indices = bin2dec(torch.from_numpy(indices), args_dict['n_bits'])
    indices = np.reshape(indices.cpu().numpy(), (len(quant_params), -1))
    indices_dict = OrderedDict()
    for i, key in enumerate(args_dict['params']):
        indices_dict[key] = indices[i]
    
    return codebook, indices_dict['ins_feat']

def calculate_iou(masks1, masks2, base=None):
    """
    Calculate the Intersection over Union (IoU) between two sets of masks.
    Args:
        masks1: PyTorch tensor of shape [n, H, W], torch.int32.
        masks2: PyTorch tensor of shape [m, H, W], torch.int32.
    Returns:
        iou_matrix: PyTorch tensor of shape [m, n], containing IoU values.
    """
    # Ensure the masks are of type torch.int32
    if masks1.dtype != torch.bool:
        masks1 = masks1.to(torch.bool)
    if masks2.dtype != torch.bool:
        masks2 = masks2.to(torch.bool)
    
    # Compute IoU
    n, H, W = masks1.shape
    m, _, _ = masks2.shape
    
    iou_matrix = torch.zeros((m, n), device=masks1.device)
    
    # Iterate over m (usually smaller, e.g. 10) to avoid OOM with large [m, n, H, W] tensor
    for i in range(m):
        mask2 = masks2[i].unsqueeze(0) # [1, H, W]
        
        # Intersection: [n]
        intersection = (masks1 & mask2).float().sum(dim=(1, 2))
        
        if base == "former":
            union = masks1.float().sum(dim=(1, 2)) + 1e-6
        elif base == "later":
            union = mask2.float().sum(dim=(1, 2)) + 1e-6
        else:
            union = (masks1 | mask2).float().sum(dim=(1, 2)) + 1e-6
            
        iou_matrix[i] = intersection / union
    
    return iou_matrix

def get_SAM_mask_and_feat(gt_sam_mask, level=3, filter_th=50, original_mask_feat=None, sample_mask=False):
    """
    input: 
        gt_sam_mask[4, H, W]: mask id
    output:
        mask_id[H, W]: The ID of the mask each pixel belongs to (0 indicates invalid pixels)
        mask_bool[num_mask+1, H, W]: Boolean, note that the return value excludes the 0th mask (invalid points)
        invalid_pix[H, W]: Boolean, invalid pixels
    """
    # (1) mask id: -1, 1, 2, 3,...
    mask_id = gt_sam_mask[level].clone()
    if level > 0:
        # subtract the maximum mask ID of the previous level
        mask_id = mask_id - (gt_sam_mask[level-1].max().detach().cpu()+1)
    if mask_id.min() < 0:
        mask_id = mask_id.clamp_min(-1)    # -1, 0~num_mask
    mask_id += 1    # 0, 1~num_mask+1
    invalid_pix = mask_id==0    # invalid pixels

    # (2) mask id[H, W] -> one-hot/mask_bool [num_mask+1, H, W]
    instance_num = mask_id.max()
    num_classes = int(instance_num.item() + 1)
    
    # [Optimization] Use loop to create boolean mask directly to save memory
    H, W = mask_id.shape
    mask_bool = torch.zeros((num_classes, H, W), dtype=torch.bool, device=mask_id.device)
    for i in range(num_classes):
        mask_bool[i] = (mask_id == i)
    
    # # TODO modify -------- only keep the largest 50
    # if instance_num > 50:
    #     top50_values, _ = torch.topk(mask_bool.sum(dim=(1,2)), 50, largest=True)
    #     filter_th = top50_values[-1].item()
    # # modify --------

    # # TODO: not used
    # # (3) delete small mask 
    # saved_idx = mask_bool.sum(dim=(1,2)) >= filter_th  # default 50 pixels
    # # Random sampling, not actually used
    # if sample_mask:
    #     prob = torch.rand(saved_idx.shape[0])
    #     sample_ind = prob > 0.5
    #     saved_idx = saved_idx & sample_ind.cuda()
    # saved_idx[0] = True  # Keep the mask for invalid points, ensuring that mask_id == 0 corresponds to invalid pixels.
    # mask_bool = mask_bool[saved_idx]    # [num_filt, H, W]

    # update mask id
    mask_id = torch.argmax(mask_bool.to(torch.int8), dim=0)  # [H, W] The ID of the pixels after filtering is 0
    invalid_pix = mask_id==0

    # TODO not used!
    # (4) Get the language features corresponding to the masks (used for 2D-3D association in the third stage)
    if original_mask_feat is not None:
        mask_feat = original_mask_feat.clone()       # [num_mask, 512]
        max_ind = int(gt_sam_mask[level].max())+1
        min_ind = int(gt_sam_mask[level-1].max())+1 if level > 0 else 0
        mask_feat = mask_feat[min_ind:max_ind, :]
        # # update mask feat
        # mask_feat = mask_feat[saved_idx[1:]]    # The 0th element of saved_idx is the mask corresponding to invalid pixels and has no features

        return mask_id, mask_bool[1:, :, :], mask_feat, invalid_pix
    return mask_id, mask_bool[1:, :, :], invalid_pix

def pair_mask_feature_mean(feat_map, masks):
    """ mean feat of N masks
    feat_map: [N, C, H, W]
    masks: [N, H, W]
    mean_values: [N, C]
    """
    N, C, H, W = feat_map.shape

    # [N, H, W] -> [N, C, H, W]
    expanded_masks = masks.unsqueeze(1).expand(-1, C, -1, -1)
    # [N, C, H, W]
    masked_features = feat_map * expanded_masks.float()
    # pixels
    mask_counts = expanded_masks.sum(dim=[2, 3]) + 1e-6
    # mean feat [N, C]
    mean_values = masked_features.sum(dim=[2, 3]) / mask_counts

    return mean_values

def mask_feature_mean(feat_map, gt_masks, image_mask=None, return_var=False):
    """Compute the average instance features within each mask.
    feat_map: [C=6, H, W]         the instance features of the entire image
    gt_masks: [num_mask, H, W]  num_mask boolean masks
    """
    num_mask, H, W = gt_masks.shape
    C = feat_map.shape[0]
    
    mean_per_channel = torch.zeros((num_mask, C), device=feat_map.device)
    mask_counts_all = torch.zeros((num_mask), device=feat_map.device)
    
    # Handle image_mask shape [1, H, W] or [H, W]
    if image_mask is not None:
        if image_mask.dim() == 3 and image_mask.shape[0] == 1:
            image_mask = image_mask.squeeze(0)
        # image_mask is now [H, W]

    chunk_size = 1 # Small chunk size to avoid OOM
    
    for i in range(0, num_mask, chunk_size):
        end_i = min(i + chunk_size, num_mask)
        chunk_masks = gt_masks[i:end_i] # [chunk, H, W]
        
        # Expand feat_map: [1, C, H, W] -> [chunk, C, H, W]
        chunk_feat = feat_map.unsqueeze(0).expand(end_i - i, -1, -1, -1)
        
        chunk_masks_expanded = chunk_masks.unsqueeze(1) # [chunk, 1, H, W]
        
        if image_mask is not None:
             # Apply image_mask to chunk_masks
             chunk_masks_expanded = chunk_masks_expanded * image_mask.unsqueeze(0).unsqueeze(0)
        
        # Compute masked features for this chunk
        # [chunk, C, H, W]
        chunk_masked_feats = chunk_feat * chunk_masks_expanded.float()
        
        # Count
        chunk_counts = chunk_masks_expanded.sum(dim=(2, 3)).squeeze(1) # [chunk]
        mask_counts_all[i:end_i] = chunk_counts
        
        # Sum
        chunk_sum = chunk_masked_feats.sum(dim=(2, 3)) # [chunk, C]
        
        # Mean
        mean_per_channel[i:end_i] = chunk_sum / chunk_counts.unsqueeze(1).clamp(min=1)
        
        del chunk_masked_feats
        del chunk_sum
        torch.cuda.empty_cache()

    if not return_var:
        return mean_per_channel

    variance_per_channel = torch.zeros((num_mask, C), device=feat_map.device)
    
    for i in range(0, num_mask, chunk_size):
        end_i = min(i + chunk_size, num_mask)
        chunk_masks = gt_masks[i:end_i]
        chunk_mean = mean_per_channel[i:end_i] # [chunk, C]
        
        chunk_feat = feat_map.unsqueeze(0).expand(end_i - i, -1, -1, -1)
        chunk_masks_expanded = chunk_masks.unsqueeze(1)
        
        if image_mask is not None:
             chunk_masks_expanded = chunk_masks_expanded * image_mask.unsqueeze(0).unsqueeze(0)

        # Variance: sum((x - mean)^2 * mask) / count
        # sum((x-mean)^2 * mask) = sum(x^2 * mask) - 2*mean*sum(x*mask) + mean^2*sum(mask)
        # But sum(x*mask) / sum(mask) = mean.
        # So sum(x*mask) = mean * count.
        # sum((x-mean)^2 * mask) = sum(x^2 * mask) - 2*mean*(mean*count) + mean^2*count
        #                        = sum(x^2 * mask) - mean^2*count
        # Var = sum(x^2 * mask)/count - mean^2
        
        chunk_sq_masked = (chunk_feat ** 2) * chunk_masks_expanded.float()
        chunk_sq_sum = chunk_sq_masked.sum(dim=(2, 3)) # [chunk, C]
        
        chunk_counts = mask_counts_all[i:end_i].unsqueeze(1).clamp(min=1)
        
        chunk_var = (chunk_sq_sum / chunk_counts) - (chunk_mean ** 2)
        # Clamp to avoid negative variance due to precision errors
        chunk_var = torch.clamp(chunk_var, min=0.0)
        
        variance_per_channel[i:end_i] = chunk_var
        
        del chunk_sq_masked
        del chunk_sq_sum
        torch.cuda.empty_cache()
        
    return mean_per_channel, variance_per_channel.mean(dim=1), mask_counts_all

def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps)**(5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb):
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps)**(12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(((200 * srgb + 11) / (211)), eps)**(12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError