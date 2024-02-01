import open3d as o3d
import numpy as np
from utils import transform_util
import copy
import random

def load_pt(pt_file, n_pts=None):
    chair = np.load(pt_file)["points"]
    print('chair dimension ' + str(chair.shape))
    if n_pts:
        idx = random.sample(range(100000), n_pts)
        chair = chair[idx] # dim: [1024, 3]
    return chair


def save_mesh_to_world(obj_file, output_obj_file, world_mean, world_scale, rotation):
    '''save mesh to world space to be visualized in Blender
    '''
    try :
        mesh_input = o3d.io.read_triangle_mesh(obj_file)
    except:
        print("input mesh has problem, terminate experiment!!")
        return False
    # apply rotation based on shapenet original obj file
    pts = np.asarray(mesh_input.sample_points_uniformly(number_of_points=100000).points)
    # normalize sampled points in a unit bbox
    norm_pt, unit_mean, unit_scale = transform_util.normalize_point_cloud(pts)
    # scale and shift mesh such that it aligns with unit point cloud
    mesh_s = copy.deepcopy(mesh_input).translate((-unit_mean[0],-unit_mean[1],-unit_mean[2]))
    mesh_s.scale(1/unit_scale, center=np.zeros(3))
    # +90: up axis in blender is not y
    R = mesh_s.get_rotation_matrix_from_xyz((0, - rotation / 180 * np.pi, 0))
    mesh_s.rotate(R, center=np.zeros(3))
    # scale and shift mesh again using provided shift and scale, which lifts it to the world scale
    mesh_world = copy.deepcopy(mesh_s)
    # bring back to world coordinates, scale before shifting
    mesh_world.scale(world_scale, center=np.zeros(3))
    mesh_world.translate((world_mean[0],world_mean[1],world_mean[2]))
    # write world mesh to obj
    o3d.io.write_triangle_mesh(output_obj_file,
                            mesh_world,
                            write_triangle_uvs=True)
    return True


def write_bvh_from_txt(filename_txt, output_bvh, bvh_header, opt_iterations):
    ''' read bvh header file and motion text file to combine into one BVH file
    '''
    # Reading bvh header
    # Reading txt motion data
    with open(filename_txt) as fp:
        motion_content = fp.read()
    write_bvh_from_motion(motion_content, output_bvh, bvh_header, opt_iterations)


def write_bvh_from_motion(motion_content, output_bvh, bvh_header, opt_iterations):
    ''' read bvh header file and concatenate with motion to combine into one BVH file

    args:
        motion_content: string with multiple lines, demiliter is white space
    '''
    with open(bvh_header) as fp:
        bvh_header = fp.read()

    # Merging 2 files
    bvh_header += "Frames: " + str(opt_iterations) + "\n"
    bvh_header += "Frame Time: 0.02\n"
    bvh_header += motion_content

    with open(output_bvh, 'w') as fp:
        fp.write(bvh_header)
