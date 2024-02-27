
import numpy as np
import torch 
import torchgeometry as tgm
from scipy.spatial.transform import Rotation
import copy

def hom(rot, pos):
    trans = np.eye(4)
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = pos
    return trans

def hom_nn(rot, pos):
    row = torch.tensor([0,0,0,1])[None, :].float().cuda()
    trans = torch.cat([rot, pos[:, None]], axis=1)
    trans = torch.cat([trans, row])
    return trans

def hom_nn_pt(pos):
    rot = torch.eye(3).float().cuda()
    row = torch.tensor([0,0,0,1])[None, :].float().cuda()
    trans = torch.cat([rot, pos[:, None]], axis=1)
    trans = torch.cat([trans, row])
    return trans


def build_tree_from_mocap(mocap, frame_index):
    # return a dictionary of joints an its dof (offset, rotation)
    tree = {}
    root_name = 'Hips'
    tree[root_name] = {}
    r_offset = torch.from_numpy(np.array(mocap.joint_offset(root_name))).float().cuda()
    tree[root_name]['offset'] = r_offset
    r_rot = frame_joint_rot(mocap, frame_index, root_name)
    tree[root_name]['rotation'] = r_rot
    tree_recur(mocap, root_name, tree[root_name], frame_index)
    return tree


def tree_recur(mocap, p_joint_name, tree, frame_index):
    children = mocap.joint_direct_children(p_joint_name)
    finger_condition = p_joint_name == 'LeftHand' or p_joint_name == 'RightHand'
    if finger_condition:
        return
    if (len(children) == 0):
        # 13 end effector
        cur_off = torch.from_numpy(np.array(get_joint_end_site_offset(mocap, p_joint_name))).float().cuda()
        tree[p_joint_name + '_End'] = {}
        tree[p_joint_name + '_End']['offset'] = cur_off
        return
    else: 
        for i in range(len(children)):
            tree[children[i].name] = {}
            cur_off = torch.from_numpy(np.array(mocap.joint_offset(children[i].name))).float().cuda()
            cur_rot = frame_joint_rot(mocap, frame_index, children[i].name) # useless
            tree[children[i].name]['offset'] = cur_off
            tree[children[i].name]['rotation'] = cur_rot
            tree_recur(mocap, children[i].name, tree[children[i].name], frame_index)


def initialize_tensor(param_dict):
    '''
    Returns:
        param_dict: a nested dict of all DoF, which is later converted to tensor parameters which require gradient
    '''
    tensor_dict = copy.deepcopy(param_dict)
    param_trans_list = []
    param_angle_list = []
    key = 'Hips'
    tensor_dict[key]['translation'] = torch.from_numpy(tensor_dict[key]['translation']).float().cuda()
    tensor_dict[key]['translation'].requires_grad_()
    tensor_dict[key]['rotation'] = torch.from_numpy(tensor_dict[key]['rotation']).float().cuda()
    tensor_dict[key]['rotation'].requires_grad_()
    param_trans_list.append(tensor_dict[key]['translation'])
    param_angle_list.append(tensor_dict[key]['rotation'])
    initialize_tensor_recur(tensor_dict[key], param_angle_list)
    return param_trans_list, param_angle_list, tensor_dict


def initialize_tensor_recur(tensor_dict, param_list):
        children = list(tensor_dict.keys())
        if 'translation' in children:
            children.remove("translation")
        if 'rotation' in children:
            children.remove("rotation")
        if (len(children) == 0):
            return
        for key in children:
            tensor_dict[key]['rotation'] = torch.from_numpy(tensor_dict[key]['rotation']).float().cuda()
            tensor_dict[key]['rotation'].requires_grad_()
            param_list.append(tensor_dict[key]['rotation'])
            initialize_tensor_recur(tensor_dict[key], param_list)


def toggle_gradient(tensor_dict, grad_list, optimize_all):
    if 'Hips' in grad_list or optimize_all:
        joint_grad = True
    else:
        joint_grad = False
    tensor_dict['Hips']['translation'].requires_grad = joint_grad
    tensor_dict['Hips']['rotation'].requires_grad = joint_grad
    toggle_gradient_recur(tensor_dict['Hips'], grad_list, optimize_all)


def toggle_gradient_recur(tensor_dict, grad_list, optimize_all):
    children = list(tensor_dict.keys())
    if 'translation' in children:
        children.remove("translation")
    if 'rotation' in children:
        children.remove("rotation")
    if (len(children) == 0):
        return
    else:
        for key in children:
            if key in grad_list or optimize_all:
                joint_grad = True
            else:
                joint_grad = False
            tensor_dict[key]['rotation'].requires_grad = joint_grad
            toggle_gradient_recur(tensor_dict[key], grad_list, optimize_all)



def forward_kinematics_nn_dict(tree, tensor_dict, save_motion=False, 
                               global_offset=torch.zeros(3), ignore_root_transform=False):

    '''A differentiable function used in optimization to compute forward kinematics
    # trans and rot are optimisable DoFs
    # trans needs to be the unnormalized
    Args:
        tree: stores the information about offset
        rot: dimension: 67, 3, 3

    '''
    global_offset = global_offset.cuda()
    motion_line = []
    rotation_dict = {}
    r_offset = tree['Hips']['offset'] + global_offset
    r_pos = tensor_dict['Hips']['translation'] + r_offset
    r_rot = tensor_dict['Hips']['rotation']
    rotation_dict["Hips"] = r_rot
    r_rot = tgm.angle_axis_to_rotation_matrix(r_rot[None, :])[0, 0:3, 0:3]

    if save_motion:
        r_rot_np = r_rot.detach().cpu().numpy()
        r_pos_np = r_pos.detach().cpu().numpy()
        motion_line.extend([r_pos_np[0], r_pos_np[1], r_pos_np[2]])
        euler = Rotation.from_matrix(r_rot_np).as_euler('zyx', degrees=True)
        motion_line.extend([euler[2], euler[1], euler[0]])
    if ignore_root_transform:
        r_m = torch.eye(4).cuda()
        r_pos = torch.zeros(3).cuda()
    else:
        r_m = hom_nn(r_rot, r_pos)
    positions = recur_nn_dict(tree['Hips'], tensor_dict['Hips'], r_m, r_pos[None, :], motion_line, save_motion, rotation_dict)
    motion_line_str = ' '.join(map(str, motion_line))
    return positions, motion_line_str, rotation_dict


def recur_nn_dict(tree, tensor_dict, p_m, positions, motion_line, save_motion, rotation_dict):
    children = list(tree.keys())
    children.remove('rotation')
    children.remove('offset')
    for i in range(len(children)):
        if 'End' in children[i]:
            cur_off = tree[children[i]]['offset']
            cur_m = hom_nn_pt(cur_off)
            cur_final = torch.matmul(p_m, cur_m)
            cur_pos = ((cur_final[0:3, 3] / cur_final[3, 3])[None, :])
            positions = torch.cat([positions, cur_pos])
        else:
            cur_off = tree[children[i]]['offset']
            cur_rot = tensor_dict[children[i]]['rotation']
            rotation_dict[children[i]] = cur_rot
            cur_rot = tgm.angle_axis_to_rotation_matrix(cur_rot[None, :])[0, 0:3, 0:3]
            if save_motion:
                euler = Rotation.from_matrix(cur_rot.detach().cpu().numpy()).as_euler('zyx', degrees=True)
                motion_line.extend([euler[2], euler[1], euler[0]])
            cur_m = hom_nn(cur_rot, cur_off)
            cur_m = torch.matmul(p_m, cur_m)
            cur_pos = (cur_m[0:3, 3] / cur_m[3, 3])[None, :]
            positions = torch.cat([positions, cur_pos])
            positions = recur_nn_dict(tree[children[i]], tensor_dict[children[i]], cur_m, positions, motion_line, save_motion, rotation_dict)
    return positions


def frame_root_pos(mocap, frame_index):
    joint = 'Hips'
    x = mocap.frame_joint_channel(frame_index, joint, 'Xposition')
    y = mocap.frame_joint_channel(frame_index, joint, 'Yposition')
    z = mocap.frame_joint_channel(frame_index, joint, 'Zposition')
    return np.array([x, y, z])

def frame_joint_rot(mocap, frame_index, joint):
    x = mocap.frame_joint_channel(frame_index, joint, 'Xrotation')/180 * np.pi
    y = mocap.frame_joint_channel(frame_index, joint, 'Yrotation')/180 * np.pi
    z = mocap.frame_joint_channel(frame_index, joint, 'Zrotation')/180 * np.pi
    rx = np.eye(3)
    rx[1, 1] = np.cos(x)
    rx[1, 2] = -np.sin(x)
    rx[2, 1] = np.sin(x)
    rx[2, 2] = np.cos(x)
    ry = np.eye(3)
    ry[0, 0] = np.cos(y)
    ry[0, 2] = np.sin(y)
    ry[2, 0] = -np.sin(y)
    ry[2, 2] = np.cos(y)
    rz = np.eye(3)
    rz[0, 0] = np.cos(z)
    rz[0, 1] = -np.sin(z)
    rz[1, 0] = np.sin(z)
    rz[1, 1] = np.cos(z)
    # in our BVH definition, rotation order is x y z
    return np.matmul(np.matmul(rx, ry), rz)
    
def get_joint_end_site_offset(mocap, joint_name):
    # used when joint has no children
    joint = mocap.get_joint(joint_name)
    end = [child for child in joint.filter('End')]
    offset = end[0]['OFFSET']
    return (float(offset[0]), float(offset[1]), float(offset[2]))


def forward_kinematics_pos_dict(mocap, frame_index, ignore_root_transform=False):
    ''''
    Returns:
        Positions: positions of each joint
        tensor_dict: a nested dict of all DoF, which is later converted to tensor parameters which require gradient
        joint_list: a list of joints in the order of recursion
    '''
    positions = []
    tensor_dict = {}
    joint_list = []
    root_name = 'Hips'
    joint_list.append(root_name)
    r_offset = np.array(mocap.joint_offset(root_name))
    trans = frame_root_pos(mocap, frame_index)
    r_pos = trans + r_offset
    r_rot = frame_joint_rot(mocap, frame_index, root_name)

    if ignore_root_transform:
        r_m = np.eye(4)
        r_pos = np.zeros(3)
        r_rot = np.eye(3)
    else:
        r_m = hom(r_rot, r_pos)
    positions.append(r_pos)
    tensor_dict[root_name] = {}
    tensor_dict[root_name]['translation'] = r_pos

    r = Rotation.from_matrix(r_rot)
    r_rot_save = r.as_rotvec()

    tensor_dict[root_name]['rotation'] = r_rot_save
    recur_pos_dict(mocap, r_m, root_name, positions, tensor_dict[root_name], frame_index, joint_list)
    return np.array(positions), tensor_dict, joint_list


def recur_pos_dict(mocap, p_m, p_joint_name, positions, tensor_dict, frame_index, joint_list):
    children = mocap.joint_direct_children(p_joint_name)
    # ignore fingers
    finger_condition = p_joint_name == 'LeftHand' or p_joint_name == 'RightHand'
    if finger_condition:
        return
    if (len(children) == 0):
        # only read offset, end effector does not have additional rotation
        joint_list.append(p_joint_name + '_End')
        cur_m = np.eye(4)
        cur_off = get_joint_end_site_offset(mocap, p_joint_name)
        cur_m[0:3, 3] = cur_off
        cur_final = np.matmul(p_m, cur_m)
        cur_pos = cur_final[0:3, 3] / cur_final[3, 3]
        positions.append(cur_pos)
        return
    else:
        for i in range(len(children)):
            joint_list.append(children[i].name)
            cur_off = mocap.joint_offset(children[i].name)
            cur_rot = frame_joint_rot(mocap, frame_index, children[i].name)
            cur_m = hom(cur_rot, cur_off)
            cur_m = np.matmul(p_m, cur_m)
            cur_pos = cur_m[0:3, 3] / cur_m[3, 3]
            positions.append(cur_pos)
            tensor_dict[children[i].name] = {}
            r = Rotation.from_matrix(cur_rot)
            cur_rot_save = r.as_rotvec()
            tensor_dict[children[i].name]['rotation'] = cur_rot_save
            recur_pos_dict(mocap, cur_m, children[i].name, positions, tensor_dict[children[i].name], frame_index, joint_list)

