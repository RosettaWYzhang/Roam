
# Setup Instruction

## Step 1: Clone the Repository
```
git clone ...
cd ROAM
```

## Step 2: Create New Conda Environment
```
conda create --name roam_env python=3.9
conda activate roam
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

## Step 3: Install Requirements
```
pip install -r requirements.txt
cd NDF
git clone https://github.com/20tab/bvh-python.git
cd bvh-python
pip install .
```

# Data Instruction

You do not need to download any data to run our Unity demo. 

If you want to play around with our Goal Pose Optimization module, please download a subset of shapenet dataset with occupancy labels.
Only the chair (ID: 03001627) and sofa (ID: 04256520) categories are used in the method.
- download preprocessed data from [occupancy network](https://github.com/autonomousvision/occupancy_networks) and rename the folder as shapenet_pointnet
- download [ShapeNet V1](https://shapenet.org/download/shapenetcore) for the corresponding meshes needed for visualization; rename the folder as shapenet_mesh
- If you want to process reference poses for NDF optimization, please download from (insert link). 
Under data/ndf_data/ref_motion, there is a blender file which contains reference poses and objects for visualization and reference pose selection. 
- If you want to re-train L-NSM using our preprocessed data, please download from (insert link) \
- If you want access to our raw motion data in BVH format, please download from (insert link) \

The folder structure should look like: 
```
ROAM/data/
         ndf_data/
                 ref_meshes/
                 ref_motion/
                 ref_objects/
                 shapenet_mesh/
                              03001627/
                              04256520/
                 shapenet_pointcloud/
                              03001627/
                              04256520/         
         l_nsm_data_processed/
         mocap_raw_data/
```


# Code 
There are three main components of our code: Goal Pose Optimization (under NDF folder), L_NSM for training the motion model and Roam_Unity for the motion inference and demo. 

## Goal Pose Optimization
### Running Optimization Using Pretrained Models:
```
cd NDF
```
#### Chair Sit 
```
python optimize_main.py --category "chair" --sequence_name "chair_sit" --shapenet_id f2e2993abf4c952b2e69a7e134f91051 --ref_pose_index 10 --novel_obj_angle -70 --exp_name "chair_demo"
```
#### Sofa Sit 
```
python optimize_main.py --category "sofa" --sequence_name "sofa_sit" --shapenet_id 824953234ed5ce864d52ab02d0953f29 --ref_pose_index 100 --novel_obj_angle 30 --exp_name "sofa_sit_demo"
```
#### Sofa Lie
```
python optimize_main.py --category "sofa" --sequence_name "sofa_lie" --shapenet_id 2e12af86321da41284e6e639680867d1 --ref_pose_index 30 --novel_obj_angle 50 --exp_name "sofa_lie_demo"
```
### Explanation of Output
To examine the output: load *.obj file and *.bvh into Blender.
In Blender, you can visualize the optimization process for 0 to 500 iterations.
If you want to see the motion result on this object and optimized pose, you need to load both .*obj and final_pose_unity.txt is loaded into the Unity project. 

### Optimize Other Poses and Objects
You can open the blender file under data/ndf_data/ref_motion to obtain the indices for other references poses.
You can select the objects according to the images in data/ndf_data/shapenet_mesh.

### Re-train NDF:
```
cd NDF/NDFNet
python train_vnn_occupancy_net.py --obj_class chair --experiment_name ndf_chair_model_opensource
python train_vnn_occupancy_net.py --obj_class sofa --experiment_name ndf_sofa_model_opensource
```

## Roam Unity
Unity version: 2021.3.14 \

Open the Demo Scene (Roam_Unity -> Assets -> Scenes -> RoamDemo.unity).

We provide both high-level and low-level modes. 

In the high-level mode, the pipeline runs automatically after hitting the Play button with randomly sampled novel objects from the test set as well as optimized poses from randomly sampled reference poses. 
This the same challenging setting which we used in the quantitative evaluation. 

To enable low-level mode, please disable the HighLevel option from the Roam_Demo interface and enable the LowLevel object collections. 
- Hit the Play button.
- Move around with W,A,S,D (Move), Q,E (Turn)
- Once close to the object, press either C for sitting down, or L for lying down. 
- Feel free to import novel objects and optimized poses from the NDF module!


### Motion Re-export
If you want to visualize the annotated data or re-export the motion, please download the [preprocessed motion asssets]() and put them under Roam_Unity/Assets/MotionCapture.

The folder structure should look like:
```
Roam_Unity/Assets/MotionCapture/
                            forward0327/
                            lie_transition_25fps/
                            lie_walk_25fps/
                            random_0327/
                            side_0327/
                            sit_sofa_transition_25fps/
                            sit_sofa_walk_25fps/
                            transition171122/
```

Open MotionProcess.Unity under the Scenes folder and it contains the motion processing interface with the annotated eight sequences which we exported for L_NSM training.

To re-export, please navigate to AI4Animation -> Motion Exporter in the menu bar. 
### Retrain L-NSM:
```
cd L_NSM
python main.py --config l_nsm.yaml
```

# Acknowledgement
Parts of our code are adapted from [AI4Animations](https://github.com/sebastianstarke/AI4Animation), 
[Couch](https://github.com/xz6014/couch/) and
[Neural Descriptor Fields](https://github.com/anthonysimeonov/ndf_robot).
We thank the authors for making the code public. 

# License
Permission is hereby granted, free of charge, to any person or company obtaining a copy of this software and associated documentation files (the "Software") from the copyright holders to use the Software for any non-commercial purpose. Publication, redistribution and (re)selling of the software, of modifications, extensions, and derivates of it, and of other software containing portions of the licensed Software, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee.

Packaging or distributing parts or whole of the provided software (including code, models and data) as is or as part of other software is prohibited. Commercial use of parts or whole of the provided software (including code, models and data) is strictly prohibited. Using the provided software for promotion of a commercial entity or product, or in any other manner which directly or indirectly results in commercial gains is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Citation
```
@article{zhang2023roam,
    title = {ROAM: Robust and Object-aware Motion Generation using Neural Pose Descriptors},
    author = {Zhang, Wanyue and Dabral, Rishabh and Leimk{\"u}hler, Thomas and Golyanik, Vladislav and Habermann, Marc and Theobalt, Christian},
    year = {2024},
    journal={International Conference on 3D Vision (3DV)}
}
```