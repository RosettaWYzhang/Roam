import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob
from scipy.spatial.transform import Rotation
import copy



class JointOccTrainDataset(Dataset):
    def __init__(self, phase='train', obj_class='chair', 
                 chair_path="../../data/ndf_data/shapenet_pointcloud/03001627", 
                 sofa_path="../../data/ndf_data/shapenet_pointcloud/04256520"):
        self.obj_class = obj_class
        # Path setup (change to folder where your training data is kept)
        ## these are the names of the full dataset folders
        paths = []

        if 'chair' in obj_class:
            paths.append(chair_path)

        if 'sofa' in obj_class:
            paths.append(sofa_path)

        print('Loading from paths: ', paths)

        files_total = []
        pt_files_total = []
        for path in paths:
         
            files = list(sorted(glob.glob(path+"/*/points.npz")))
            pt_files = list(sorted(glob.glob(path+"/*/pointcloud.npz")))

            n = len(files)
            idx = int(0.9 * n)

            if phase == 'train':
                files = files[:idx]
                pt_files = pt_files[:idx]
            else:
                files = files[idx:]
                pt_files = pt_files[idx:]
            files_total.extend(files)
            pt_files_total.extend(pt_files)
            self.pt_files = pt_files_total
                

        self.files = files_total
        # perturbation parameters
        block = 128 
        bs = 1 / block
        hbs = bs * 0.5
        self.bs = bs
        self.hbs = hbs
        print("files length ", len(self.files))


    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        try:
            data = np.load(self.files[index], allow_pickle=True)
            pt_data = np.load(self.pt_files[index], allow_pickle=True)

            # chair/sofa category
            voxel_bool = np.unpackbits(data['occupancies'])
            coord = data['points']
            point_cloud = pt_data['points'] 
            #point_cloud = rotate_centered_pointcloud(point_cloud, random.uniform(0, 360))

            rix = np.random.permutation(coord.shape[0])
            coord = coord[rix[:1500]]
            labels = np.expand_dims(voxel_bool[rix[:1500]], -1)

            # normalization of coord
            # divided into 128 voxels, the perturbation is only within the voxel
            # hbs: 1/128 * 0.5
            offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord + offset # adding noise within the grid 
            coord = coord * data['scale'] # offset first, then scale
            labels = (labels - 0.5) * 2.0 # normalization of label
            
            rix = torch.randperm(point_cloud.shape[0])
            point_cloud = point_cloud[rix[:1500]]
            res = {
                    'point_cloud': point_cloud.astype(np.float32),
                    'coords': coord.astype(np.float32),
                    'intrinsics': np.zeros([3, 4]).astype(np.float32),
                    'cam_poses': np.zeros(1).astype(np.float32)}

            return res, {'occ': torch.from_numpy(labels).float()}

        except Exception as e:
           print(e)
           return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)


def normalize_point_cloud(pt):
    '''' this function normalize the point cloud in the same way as OccupancyNetwork
    3D bounding box of the mesh is centered at 0 and its longest edge has a length of 1

    Returns:
        pt_norm: normalized point clouds
        ref_mean: translation to center the point cloud
        scale: length of longest side of the bounding box
    '''
    pt_norm = copy.deepcopy(pt)
    bb_min = np.min(pt, axis=0)[1]
    bb_max = np.max(pt, axis=0)[1]
    ref_mean = (bb_max + bb_min)/2
    scale = np.max(bb_max - bb_min)
    pt_norm_y = (pt[:, 1] - ref_mean) / scale
    pt_norm[:, 1] = pt_norm_y
    return pt_norm, ref_mean, scale


def rotate_centered_pointcloud(pt, angle_in_degree=0):
    '''
    this function assumes that point cloud is 0-centered
    '''
    perturb_matrix = Rotation.from_euler('y', angle_in_degree, degrees=True).as_matrix()
    pt = np.matmul(pt, perturb_matrix)
    return pt