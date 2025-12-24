import torch
import numpy as np
from sklearn.cluster import Birch

class GraphCluster:
    def __init__(self, feat_scale=1.0, branching_factor=50, threshold=0.5):
        self.feat_scale = feat_scale
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.model = None
        self.node_centers = None # Store cluster centers (Super-Gaussians)
        self.point_labels = None # Cluster ID for each Gaussian point
        self.node_features = None

    def fit(self, gaussians):
        """
        Execute clustering logic:
        1. Extract features f = [In, XYZ]
        2. Use BIRCH for hierarchical clustering
        """
        print("Starting Graph Clustering (BIRCH)...")
        
        # 1. Prepare features: Concatenate instance features (In) and spatial coordinates (XYZ)
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        # Use get_ins_feat(origin=True) to get the continuous features
        features = gaussians.get_ins_feat(origin=True).detach().cpu().numpy()
        
        # Normalize or scale features
        # Note: features are usually 6-dim or similar.
        data = np.concatenate([features * self.feat_scale, xyz], axis=1)
        
        # 2. Initialize and run BIRCH
        # threshold: determines the radius of the subcluster
        self.model = Birch(threshold=self.threshold, branching_factor=self.branching_factor, n_clusters=None)
        self.model.fit(data)
        
        # 3. Get results
        # labels: Cluster index for each point L = {L1, ... LM}
        self.point_labels = torch.tensor(self.model.labels_, dtype=torch.long, device=gaussians.get_xyz.device)
        
        # subcluster_centers_: Cluster centers (Super-Gaussians initial position)
        # BIRCH subcluster_centers_ contains [Feature, XYZ]
        centers = torch.tensor(self.model.subcluster_centers_, dtype=torch.float32, device=gaussians.get_xyz.device)
        self.node_centers = centers[:, -3:] # Last 3 are XYZ
        self.node_features = centers[:, :-3] # First part is features
        
        num_clusters = len(self.node_centers)
        print(f"Clustering finished. Created {num_clusters} Super-Gaussians (Nodes).")
        
        return self.point_labels, self.node_centers

    def get_quantized_features(self, gaussians):
        """
        Get quantized features (for Compactness Loss)
        i.e., each point uses its cluster center's feature instead of its own
        """
        if self.point_labels is None:
            return gaussians.get_ins_feat(origin=True)
            
        # valid_mask handles potential noise points (label -1)
        valid_mask = self.point_labels != -1
        quantized_features = torch.zeros_like(gaussians.get_ins_feat(origin=True))
        
        # Only clustered points get the cluster center feature
        indices = self.point_labels[valid_mask]
        quantized_features[valid_mask] = self.node_features[indices]
        
        # For noise points, keep original features
        quantized_features[~valid_mask] = gaussians.get_ins_feat(origin=True)[~valid_mask]
        
        return quantized_features
