import numpy as np
from bvh import Bvh
import kinematic_chain as kc


class Skeleton:

    def __init__(self, config, ref_mean, ref_scale):
        self.config = config
        with open(self.config.ref_motion_path) as f:
            self.mocap = Bvh(f.read())
        
        self.bone_dict = {"LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", 
                          "LeftForeArm" :"LeftArm", "LeftHand": "LeftForeArm",
                          "RightLeg": "RightUpLeg", "RightFoot": "RightLeg", 
                          "RightForeArm" :"RightArm", "RightHand": "RightForeArm"}
        
        self.get_ref_pos_from_bvh()
        self.positions = self.normalize_pose(self.positions_world, ref_mean, ref_scale)
        self.canonical_positions = self.normalize_pose(self.canonical_positions_world, ref_mean, ref_scale)
        self.offset_tree = self.create_offset_dof_tree()

    def get_bone_parent(self, bone_name):
        '''harded coded to add extra query points between joints
        '''
        if bone_name in self.bone_dict:
            return self.bone_dict[bone_name]
        else:
            return None

    def get_child_parent_index(self, joint_list):
        bone_index_dict = {}
        for k in self.bone_dict.keys():
            child = joint_list.index(k)
            parent = joint_list.index(self.bone_dict[k])
            bone_index_dict[child] = parent
        return bone_index_dict
    
    def get_ref_pos_from_bvh(self):
        '''
        Returns:
            positions: unnormalized according to reference mean and scale
        '''
        self.positions_world, self.param_dict, self.joint_list = kc.forward_kinematics_pos_dict(self.mocap, self.config.ref_pose_index)   
        self.canonical_positions_world, _, _ = kc.forward_kinematics_pos_dict(self.mocap, self.config.ref_pose_index, ignore_root_transform=True)
        self.num_joints = self.positions_world.shape[0]
        self.bone_index_dict = self.get_child_parent_index(self.joint_list)

    def normalize_pose(self, pose, ref_mean, ref_scale):
        pose = (pose - ref_mean) / ref_scale
        return pose
    
    def create_offset_dof_tree(self):
        # offset_tree is used in forward kinematics computation
        return kc.build_tree_from_mocap(self.mocap, self.config.ref_pose_index)  


    def initialize_skeleton(self, positions, new_scale, chair):
        ''' updating hip translation such that it's projected down to chairs in the world space

        Arguments:
            positions: normalized according to novel chair mean and scale
            new_scale: scale of the new chair
            chair: chair point cloud

        Returns:
            positions_perturbed: joint positions projected to novel chair
        '''
        hip_pos = positions[0]
        project_point, distance = self.project_to_closest_point(hip_pos, chair)
        init_shift = project_point - hip_pos
        positions_perturbed = init_shift + positions
        # record the shift in world space
        self.param_dict['Hips']['translation'] = self.param_dict['Hips']['translation'] + (init_shift * new_scale)
        return positions_perturbed
    

    def project_to_closest_point(self, point, point_cloud):
        """
        Args:
            point: 1x3 array, hip position
            point_cloud: Nx3 array, chair point cloud

        Returns: project hip position to the nearest surface point in a point cloud
            if there are surface points both above and below, prefer above point
        """ 
        eps = 0.1
        x_min = point[0] - eps
        x_max = point[0] + eps
        z_min = point[2] - eps
        z_max = point[2] + eps
        # in normalized space
        filtered_pts = [p for p in point_cloud if p[0]>x_min and p[0]<x_max and p[2]>z_min and p[2]<z_max and p[1] >= point[1]] 
        if len(filtered_pts) == 0:
            filtered_pts = [p for p in point_cloud if p[0]>x_min and p[0]<x_max and p[2]>z_min and p[2]<z_max and p[1] < point[1]] 
        point_cloud = filtered_pts
        if len(point_cloud) != 0:
            # compute the distances between each of the points and the desired point
            distances = [np.linalg.norm(p) for p in (point_cloud - point)]
            # get the index of the smalles value in the array to determine which point is closest and return it
            idx_of_min = np.argmin(distances)
            new_point = point_cloud[idx_of_min]
            return (new_point, distances[idx_of_min])
        else:
            return point, 0
        