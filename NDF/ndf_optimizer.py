import torch
import numpy as np
import kinematic_chain as kc
import torchgeometry as tgm
import pytorch3d.ops
import pickle


class Ndf_optimizer:
    def __init__(self, config, skel, model, new_mean, new_scale, reference_model_input, new_model_input, positions_init):

        self.config = config
        self.skel = skel
        self.model = model
        self.constraints_dict = self.get_constraint_dict(config)
        self.positions_init = positions_init
        

        self.l1_loss_fn = torch.nn.L1Loss()
        self.l2_loss_fn = torch.nn.MSELoss() # regularization

        model_trans_param, model_angle_param, self.tensor_dict = kc.initialize_tensor(self.skel.param_dict)
        self.new_scale_cuda = torch.tensor(new_scale).cuda().float()
        self.new_mean_cuda = torch.tensor(new_mean).cuda().float()
        self.canonical_world_tensor = (torch.from_numpy(skel.canonical_positions).float().cuda())* self.new_scale_cuda + self.new_mean_cuda
        self.opt = torch.optim.Adam([
            {'params':model_angle_param},
            {'params':model_trans_param, 'lr':self.config.lr_trans}], lr=self.config.lr) 
        self.grad_list = ["Hips"] 
        kc.toggle_gradient(self.tensor_dict, self.grad_list, optimize_all=False)  # only calculate gradient for hips DoF initially
        self.initialize_ndf_features(reference_model_input, new_model_input)


    def get_constraint_dict(self, config):
        with open(config.constraint_dict, 'rb') as f:
            constraints_dict = pickle.load(f)
        print("load constraint dict..." + config.constraint_dict)
        for k in constraints_dict.keys():
            constraints_dict[k]["lb"] -= config.relax_degree
            constraints_dict[k]["lb"] = torch.from_numpy(constraints_dict[k]["lb"]).cuda().float()
            constraints_dict[k]["ub"] += config.relax_degree
            constraints_dict[k]["ub"] = torch.from_numpy(constraints_dict[k]["ub"]).cuda().float()
        return constraints_dict


    def initialize_ndf_features(self, reference_model_input, new_model_input):
        # create query points, they are fixed for all joints and re-use for new chair
        self.query_pts = np.random.normal(0.0, self.config.sigma, size=(self.config.query_pts_per_joint, 3)).astype(float)
        # ref_query is a dictionary of joint name and joint query point cloud
        ref_query = self.create_query_point_clouds_from_joints(self.skel.positions, self.query_pts)
        new_query = self.create_query_point_clouds_from_joints(self.positions_init, self.query_pts)    
        self.ref_latent, self.ref_features = self.get_descriptor_from_query_pts(reference_model_input, ref_query)
        self.new_latent, self.new_features = self.get_descriptor_from_query_pts(new_model_input, new_query)


    def get_descriptor_from_query_pts(self, input, query_pt):
        latent = self.model.extract_latent(input).detach()  # ([1, 256, 3])
        descriptors = {}
        for j in self.skel.joint_list:
            descriptors[j] = self.model.forward_latent(latent, query_pt[j]).detach()  # ([1, ?, 2049])
        return latent, descriptors   
    

    def create_query_point_clouds_from_joints(self, positions, query_pts):
        ''' put the query points at every joint location in the point cloud
        args:
            positions: position of joints of dimension [N, 3] (normalized)

        returns:
            query_dict: a dictionary with key joint_name and value (cuda_tensor) [1, query_pts_per_joint, 3]
        '''
        query_dict = {}
        query_points = []
        for i in range(self.skel.num_joints):
            query_cloud = torch.from_numpy(positions[i, :] + query_pts).float().cuda()
            if self.config.bone_point:
                parent_bone = self.skel.get_bone_parent(self.skel.joint_list[i])
                if parent_bone is not None:
                    parent_point = query_dict[parent_bone].squeeze()[0:50, :]
                    middle_point = (parent_point + query_cloud)/2
                    query_cloud = torch.cat((query_cloud, middle_point), dim=0)     
            query_dict[self.skel.joint_list[i]] = query_cloud[None, :, :]
            query_points.append(query_dict[self.skel.joint_list[i]])      
        return query_dict
    

    def convert_joint_list_to_query_dict(self, X_new):
        X_new_dict = {}
        X_new_concat = []
        for j in range(self.skel.num_joints):
            bone_name = self.skel.joint_list[j]
            X_new_dict[bone_name] = X_new[j]
            if self.config.bone_point:
                # append 8 additional bone point cloud
                parent_bone = self.skel.get_bone_parent(self.skel.joint_list[j])
                if parent_bone is not None:
                    parent_index = self.skel.bone_index_dict[j]
                    new_pos_add = (X_new[j] + X_new[parent_index][0:50, :])/2
                    X_new_dict[bone_name] = torch.cat((X_new_dict[bone_name], new_pos_add), dim=0)
            X_new_concat.append(X_new_dict[bone_name])
        return X_new_dict

    def joint_constraint_loss(self, euler_angle, lb, ub):
        loss = torch.tensor(0).cuda().float()
        for i in range(3):
            if (euler_angle[i] < lb[i]):
                loss += (euler_angle[i] - lb[i])**2
            elif (euler_angle[i] > ub[i]):
                loss += (ub[i] - euler_angle[i])**2
        return loss 

    def joint_floor_loss(self, joint_y):
        # this loss penalize < floor level joint positions
        negative_positions = joint_y[joint_y < 0]
        loss = (negative_positions**2).sum()  
        return loss

    def reset_loss_weight(self, loss_weight):
        ''''set weight = 1 for every joint
        '''
        for k in loss_weight.keys():
            loss_weight[k] = 1
    
    def optimize(self):
        level = 0
        with open(self.config.filename_txt, 'a+') as f:
            for i in range(self.config.opt_iterations):
                if ((level < len(self.config.opt_schedule)-1 and i == self.config.opt_schedule[level + 1]) 
                    or self.config.all_stage):
                    level += 1
                    print("****** Entered LEVEL %d of optimization ********" %level)
                    if level == 1:
                        print("***** Level 1: Optimize root *****")
                    if level == 2:
                        self.reset_loss_weight(self.config.loss_weight)
                        print("***** Level 2: Optimize all joints *****") 
                        kc.toggle_gradient(self.tensor_dict, self.grad_list, optimize_all=True)
                    if level == 3:
                        print("***** Level 3: Add canonical loss and angle constraint *****")


                new_pos, motion_line, _ = kc.forward_kinematics_nn_dict(
                    self.skel.offset_tree, self.tensor_dict, save_motion=True)
                
                # add floor penetration loss before normalization
                floor_loss = torch.tensor(0).cuda().float()
                if i >= self.config.opt_schedule[-1] and self.config.use_floor_loss:
                    floor_loss = self.joint_floor_loss(new_pos[:, 1]) * self.config.floor_weight

                new_pos = (new_pos - self.new_mean_cuda) / self.new_scale_cuda 

                # penelized the distance between ref canonical pose and optimized canonical pose without root
                canonical_loss = torch.tensor(0).cuda().float()
                if i >= self.config.opt_schedule[-1] and self.config.use_canonical_loss: 
                    new_pos_canonical, _, rotation_dict = kc.forward_kinematics_nn_dict(self.skel.offset_tree, 
                                                                                        self.tensor_dict, save_motion=False,
                                                                                        ignore_root_transform=True)
                    new_pos_canonical = (new_pos_canonical - self.new_mean_cuda) / self.new_scale_cuda
                    canonical_positions_normalized = (self.canonical_world_tensor - self.new_mean_cuda) / self.new_scale_cuda
                    unweighted_canonical_loss = self.l2_loss_fn(new_pos_canonical, canonical_positions_normalized)
                    canonical_loss = self.config.canonical_loss_weight * unweighted_canonical_loss

                else:
                    unweighted_canonical_loss = torch.tensor(0).cuda().float()
                

                X_new = new_pos[:, None, :].repeat(1, self.config.query_pts_per_joint, 1) + \
                    torch.from_numpy(self.query_pts).repeat(self.skel.num_joints, 1, 1).cuda().float()  # num_joints, n_opt_pts, 3

                X_new_dict = self.convert_joint_list_to_query_dict(X_new)
                
                # X_new dim: ([num_joint, query_joint, 3])
                joint_loss_all = 0
                joint_constraint_loss_all = torch.tensor(0).cuda().float()
                if self.config.use_angle_constraint and not self.config.use_canonical_loss:
                    _, _, rotation_dict = kc.forward_kinematics_nn_dict(
                        self.skel.offset_tree, self.tensor_dict, save_motion=False, ignore_root_transform=True)

                for _, j in enumerate(self.skel.joint_list):
                    # num_joints, n_opt_pts, 3
                    self.new_features[j] = self.model.forward_latent(self.new_latent, X_new_dict[j][None, :, :])     
                    # add angle constraints
                    joint_loss = self.l1_loss_fn(self.new_features[j], self.ref_features[j]) * self.config.loss_weight[j]
                    if i >= self.config.opt_schedule[-1] and self.config.use_angle_constraint and not 'End' in j:
                        lb = self.constraints_dict[j]["lb"]
                        ub = self.constraints_dict[j]["ub"]
                        angle_matrix = tgm.angle_axis_to_rotation_matrix(rotation_dict[j].reshape(1, 3))
                        angle_matrix = angle_matrix[:, 0:3, 0:3]
                        euler_angle = torch.squeeze(tgm.rad2deg(pytorch3d.transforms.matrix_to_euler_angles(angle_matrix, "XYZ")))
                        joint_angle_loss = self.joint_constraint_loss(euler_angle, lb, ub)    
                        joint_constraint_loss_all += joint_angle_loss
                    joint_loss_all += joint_loss
                    
                joint_avg_loss = joint_loss_all / self.skel.num_joints
                constrain_avg_loss = joint_constraint_loss_all / self.skel.num_joints * self.config.constrain_weight
                loss = joint_avg_loss + canonical_loss + constrain_avg_loss + floor_loss

                print('iteration ' + str(i) + "  , total loss is " + str(loss.detach().cpu().numpy()))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                f.write(motion_line + "\n")

                
        return new_pos, self.tensor_dict, unweighted_canonical_loss


