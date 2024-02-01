
import os

class NDF_Config:

    def __init__(self, category, sequence_name, shapenet_id, ref_pose_index, novel_obj_angle, exp_name):
        # ********************************** data loading and saving *****************************************
        self.exp_name = exp_name 
        self.log = False # set to true to print to log file instead of terminal
        self.dev = 'cuda'
        self.category = category
        self.sequence_name = sequence_name # chose from "chair_sit", "sofa_sit", "sofa_lie"
        self.save_dir = os.path.join("out/", self.sequence_name, self.exp_name)
        # output filenames
        self.filename_txt = os.path.join(self.save_dir, self.exp_name + '.txt')
        self.filename_bvh = os.path.join(self.save_dir, self.exp_name + '.bvh')
        self.output_obj_file = os.path.join(self.save_dir, self.exp_name + '_mesh_world.obj')
        self.shapenet_id = shapenet_id
        model_root = "../pretrained_models/ndf/"
        self.model_path = model_root + "%s_model_final.pth" %category
        ref_root = "../data/ndf_data/"
        self.ref_obj_pt = os.path.join(ref_root, "ref_objects", "ref_%s_pointcloud.npz" %sequence_name)
        self.bvh_header_file = os.path.join(ref_root, "ref_motion", "%s_bvh_header.bvh" %sequence_name)
        self.ref_motion_path = os.path.join(ref_root, "ref_motion", "%s_motion.bvh" %sequence_name)
        shapenet_id_dict = {"chair": "03001627", "sofa": "04256520"}
        self.novel_obj_mesh = os.path.join(ref_root, "shapenet_mesh", shapenet_id_dict[self.category], 
                                           self.shapenet_id, "model.obj")
        self.novel_obj_pt = os.path.join(ref_root, "shapenet_pointcloud", shapenet_id_dict[self.category], 
                                         self.shapenet_id, "pointcloud.npz")
        
        # ********************************** data preprocessing *****************************************
        self.ref_pose_index = ref_pose_index  
        self.chair_scale = 1.0 # change this to apply additional scale to the chair
        self.novel_obj_angle = novel_obj_angle # change this to test different orientations
        
        # ********************************** optimization *****************************************
        
        self.opt_iterations = 500
        # loss weight includes end effector, 29 joints
        self.loss_weight = {'Hips': 1,
                       'Spine': 1, 'LeftUpLeg': 1, 'RightUpLeg': 1,
                       'Spine1': 1, 'LeftLeg': 3, 'RightLeg': 3,
                       'Spine2': 1, 'LeftFoot': 1, 'RightFoot': 1,
                       'Spine3': 1, 'LeftToeBase': 1, 'RightToeBase': 1,
                       'Spine4': 1, 'LeftToeBase_End': 1, 'RightToeBase_End': 1, 
                       'LeftShoulder': 1, 'RightShoulder': 1,
                       'Neck': 1, 'LeftArm': 1, 'RightArm': 1,
                       'Head': 1, 'LeftForeArm': 1, 'RightForeArm': 1,
                       'Head_End': 1, 'LeftHand': 1, 'RightHand': 1,
                       'avg_loss': 1}
        
        self.opt_schedule = [0, 150, 250, 350] # stages of optimization 
        self.n_pts = 1500  # number of points to sample from shape point cloud
        self.query_pts_per_joint = 50  # number of query points per joint
        self.sigma = 0.025 
        
        # lr 
        self.lr = 1e-2
        self.lr_trans = 1

        # pose regularization
        self.use_canonical_loss = True
        self.canonical_loss_weight = 1e-2
       
        # angle constraints
        self.use_angle_constraint = True
        self.constraint_dict = 'stats/constaints_stats_6seq.pkl'
        self.relax_degree = 10
        self.constrain_weight = 1e-3
        
        # floor penetration
        self.floor_weight = 1e-8
        self.use_floor_loss = True
        self.bone_point = True

        self.all_stage = False # set to true for ablation study
        
        