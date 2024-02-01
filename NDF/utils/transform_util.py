import numpy as np
from scipy.spatial.transform import Rotation
import random
import copy

def rotate_centered_pointcloud(pt, angle_in_degree=0):
    '''
    caution! this function assumes that point cloud is 0-centered
    '''
    if angle_in_degree == 0:
        return pt
    else:
        perturb_matrix = Rotation.from_euler('y', angle_in_degree, degrees=True).as_matrix()
        pt = np.matmul(pt, perturb_matrix)
        return pt
    

def align_ground(ref_chair_unnormalized, new_chair_world):
    # align new chair with reference chair in world space
    ref_y_min = np.min(ref_chair_unnormalized, axis=0)[1]
    new_y_min = np.min(new_chair_world, axis=0)[1]
    y_shift = ref_y_min - new_y_min
    new_chair_world = new_chair_world +  np.array([0, y_shift, 0])
    return new_chair_world


def preprocess_novel_chair(config, new_chair_dense, ref_scale, ref_mean, ref_chair_unnormalized):
    new_scale = ref_scale * config.chair_scale
    new_chair_world = new_chair_dense * new_scale + ref_mean
    new_chair_world = align_ground(ref_chair_unnormalized, new_chair_world)
    new_chair_dense, new_mean, new_scale = normalize_point_cloud(new_chair_world)
    new_chair_dense = rotate_centered_pointcloud(new_chair_dense, config.novel_obj_angle)
    idx = random.sample(range(100000), config.n_pts)
    new_chair_sparse = new_chair_dense[idx]
    return new_chair_sparse, new_mean, new_scale


def normalize_point_cloud(pt, scale_y_only=False):
    '''' this function normalize the point cloud in the same way as OccupancyNetwork
    3D bounding box of the mesh is centered at 0 and its longest edge has a length of 1

    Returns:
        pt_norm: normalized point clouds
        ref_mean: translation to center the point cloud
        scale: length of longest side of the bounding box
    '''
    if not scale_y_only:
        bb_min = np.min(pt, axis=0)
        bb_max = np.max(pt, axis=0)
        ref_mean = (bb_max + bb_min)/2
        scale = np.max(bb_max - bb_min)
        pt_norm = (pt - ref_mean) / scale
    else:
        pt_norm = copy.deepcopy(pt)
        bb_min = np.min(pt, axis=0)[1]
        bb_max = np.max(pt, axis=0)[1]
        ref_mean = (bb_max + bb_min)/2
        scale = np.max(bb_max - bb_min)
        pt_norm_y = (pt[:, 1] - ref_mean) / scale
        pt_norm[:, 1] = pt_norm_y
    return pt_norm, ref_mean, scale