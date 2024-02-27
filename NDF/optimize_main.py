
import random
import pickle
import torch
import sys
import os
import numpy as np
from pprint import pprint
from skeleton import Skeleton
from ndf_optimizer import Ndf_optimizer
from scipy.spatial.transform import Rotation
import NDFNet.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from config import NDF_Config
from utils import transform_util, io_util
from pytorch3d.loss import chamfer_distance
import argparse

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.set_default_dtype(torch.float32)


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default="chair")
parser.add_argument('--sequence_name', type=str, default="chair_sit", help="choose from <chair_sit>,  <sofa_sit>, <sofa_lie>")
parser.add_argument('--shapenet_id', type=str, default="f2e2993abf4c952b2e69a7e134f91051")
parser.add_argument('--ref_pose_index', type=int, default=10)
parser.add_argument('--novel_obj_angle', type=float, default=30)
parser.add_argument('--exp_name', type=str, default="")

args = parser.parse_args()
if args.exp_name == "":
    args.exp_name = args.shapenet_id + "_pose" + str(args.ref_pose_index) + "_angle" + str(args.novel_obj_angle)
config = NDF_Config(args.category, args.sequence_name, args.shapenet_id, args.ref_pose_index, args.novel_obj_angle, args.exp_name)

# ********************************** HELPER FUNCTIONS *****************************************
def load_ndf_model(model_path, dev):
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type="pointnet", 
                                            return_features=True,
                                            sigmoid=True).cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.to(dev)
    model.eval()
    return model


def create_ndf_model_input(chair_pt):
    model_input = {}
    shape_pcd = torch.from_numpy(chair_pt[:config.n_pts]).float().to(config.dev)
    model_input['point_cloud'] = shape_pcd[None, :, :]
    return model_input

def getAngleDiff(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2 
    return np.arccos(cos_theta) * (180/np.pi)
    
# ********************************** HELPER FUNCTIONS END *****************************************


def optimize_one_pose():
    #################  create experiment folder  #################
    if not os.path.exists(config.save_dir):
        print('creating new directory ' + config.save_dir)
        os.makedirs(config.save_dir)

    # direct print to log file 
    if config.log:
        old_stdout = sys.stdout
        log_file = open(os.path.join(config.save_dir,"out.log"), "w")
        sys.stdout = log_file
    pprint(config.__dict__)


    # ********************************** LOAD POINT CLOUDS AND NORMALIZE *****************************************
    # load chair point clouds
    ref_chair_unnormalized = io_util.load_pt(config.ref_obj_pt, config.n_pts)  
    reference_chair, ref_mean, scale = transform_util.normalize_point_cloud(ref_chair_unnormalized)
    # load novel chair point clouds
    new_chair_dense = io_util.load_pt(config.novel_obj_pt)
    new_chair, new_mean, new_scale = transform_util.preprocess_novel_chair(config, new_chair_dense, 
                                                                           scale, ref_mean, ref_chair_unnormalized)
    new_chair_world = new_chair * new_scale + new_mean
    new_mean_cuda = torch.tensor(new_mean).cuda().float()

    # compute chamfer distance in global space
    ref_world_tensor = torch.from_numpy(ref_chair_unnormalized).float().to(config.dev)[None, :, :]
    new_chair_sparse_world_tensor = torch.from_numpy(new_chair_world).float().to(config.dev)[None, :, :]
    chamfer_dist_world, _ = chamfer_distance(ref_world_tensor, new_chair_sparse_world_tensor)
    chamfer_dist_world = chamfer_dist_world.detach().cpu().item()
    
    # save mesh to world space
    io_util.save_mesh_to_world(config.novel_obj_mesh, config.output_obj_file, new_mean, new_scale, config.novel_obj_angle)

    # ******************************************* Initialize skeleton *****************************************************
    skel = Skeleton(config, ref_mean, scale)
    initial_trans = skel.param_dict["Hips"]["translation"]
    initial_rotmatrix = Rotation.from_rotvec(skel.param_dict["Hips"]["rotation"]).as_matrix()
    positions_init = (skel.positions_world - new_mean) / new_scale
    positions_init = skel.initialize_skeleton(positions_init, new_scale, new_chair)

    # ********************************** set up NDF model input *****************************************
    reference_model_input = create_ndf_model_input(reference_chair)
    new_model_input = create_ndf_model_input(new_chair)
    # compute chamfer distance on normalized chairs
    chamfer_dist, _ = chamfer_distance(reference_model_input['point_cloud'], new_model_input['point_cloud'])
    chamfer_dist = chamfer_dist.detach().cpu().item()

    
    # ********************************** INITIALIZE NDF MODEL *****************************************
    model = load_ndf_model(config.model_path, config.dev)
   
    # ********************************** OPTIMIZATION *****************************************
    optimizer = Ndf_optimizer(config, skel, model, new_mean, new_scale, reference_model_input, new_model_input, positions_init)
    new_pos, tensor_dict, unweighted_canonical_loss = optimizer.optimize()
    
    # ********************************** VISUALIZATION **************************************************
    precise_joints_unity = (((new_pos.detach().cpu().numpy()) * new_scale + new_mean) / 1000).reshape((1, -1))  # since unity is in meter
    np.savetxt(os.path.join(config.save_dir, "final_pose_unity.txt"), precise_joints_unity)

    # save the optimization process
    io_util.write_bvh_from_txt(config.filename_txt, config.filename_bvh, config.bvh_header_file, config.opt_iterations)

    # save a dictionary of statistics
    final_trans = tensor_dict["Hips"]["translation"].detach().cpu().numpy()
    final_rotation = Rotation.from_rotvec(tensor_dict["Hips"]["rotation"].detach().cpu().numpy()).as_matrix()
    translation_change = np.linalg.norm(final_trans-initial_trans)
    change_angle = getAngleDiff(final_rotation, initial_rotmatrix)
    pose_change = unweighted_canonical_loss.cpu().detach().numpy()
    world_pose_change = pose_change * new_scale
    print("############## Stats! ##########")
    print("chamfer distance is : " + str(chamfer_dist))
    print("chamfer distance world is : " + str(chamfer_dist_world))
    print("Initial translation is " + str(initial_trans))
    print("Final translation is " + str(final_trans))
    print("Change in translation is " + str(translation_change))
    print("Change in angle is " + str(change_angle))
    print("Normalized change in local pose is " + str(pose_change))
    print("World change in local pose is " + str(world_pose_change))

    chamfer_dict = {"chamfer": chamfer_dist, "chamfer_world": chamfer_dist_world,
                    "pose_distance": pose_change, "pose_distance_world": world_pose_change, 
                    "mean": new_mean_cuda.detach().cpu().numpy(),
                    "initial_rotation":initial_rotmatrix, "final_rotation": final_rotation, "rotation_change": change_angle,
                    "initial_translation": initial_trans, "final_translation": final_trans, "translation_change": translation_change,
                    }

    with open(os.path.join(config.save_dir, 'chamfer_dict.pkl'), 'wb') as fp:
        pickle.dump(chamfer_dict, fp)
        print('dictionary saved successfully to file')

    #####################################  restore standout  #################################
    if config.log:
        sys.stdout = old_stdout
        log_file.close()

def main():
    optimize_one_pose()


if __name__ == "__main__":
    main()