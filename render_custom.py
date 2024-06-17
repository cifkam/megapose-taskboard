import blenderproc as bproc
import argparse
import os
import numpy as np
from pathlib import Path
#from load_custom_objs import load_custom_objs

import os
from random import choice
from typing import List, Optional, Tuple
import warnings

import bpy
import numpy as np
from mathutils import Matrix, Vector

from blenderproc.python.utility.SetupUtility import SetupUtility
from blenderproc.python.camera import CameraUtility
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.MathUtility import change_source_coordinate_frame_of_transformation_matrix
from blenderproc.python.loader.ObjectLoader import load_obj
from blenderproc.python.loader.BopLoader import _BopLoader

def load_custom_objs(bop_dataset_path: str, model_type: str = "", obj_ids: Optional[List[int]] = None,
                  sample_objects: bool = False, num_of_objs_to_sample: Optional[int] = None,
                  obj_instances_limit: int = -1, mm2m: Optional[bool] = None, object_model_unit: str = 'm',
                  move_origin_to_x_y_plane: bool = False) -> List[MeshObject]:
    """ Loads all or a subset of 3D models of any BOP dataset

    :param bop_dataset_path: Full path to a specific bop dataset e.g. /home/user/bop/tless.
    :param model_type: Optionally, specify type of BOP model. Available: [reconst, cad or eval].
    :param obj_ids: List of object ids to load. Default: [] (load all objects from the given BOP dataset)
    :param sample_objects: Toggles object sampling from the specified dataset.
    :param num_of_objs_to_sample: Amount of objects to sample from the specified dataset. If this amount is bigger
                                  than the dataset actually contains, then all objects will be loaded.
    :param obj_instances_limit: Limits the amount of object copies when sampling. Default: -1 (no limit).
    :param mm2m: Specify whether to convert poses and models to meters (deprecated).
    :param object_model_unit: The unit the object model is in. Object model will be scaled to meters. This does not
                              affect the annotation units. Available: ['m', 'dm', 'cm', 'mm'].
    :param move_origin_to_x_y_plane: Move center of the object to the lower side of the object, this will not work
                                     when used in combination with pose estimation tasks! This is designed for the
                                     use-case where BOP objects are used as filler objects in the background.
    :return: The list of loaded mesh objects.
    """

    bop_path, bop_dataset_name = _BopLoader.setup_bop_toolkit(bop_dataset_path)

    # This import is done inside to avoid having the requirement that BlenderProc depends on the bop_toolkit
    # pylint: disable=import-outside-toplevel
    from bop_toolkit_lib import dataset_params

    # pylint: enable=import-outside-toplevel

    #########################x
    #model_p = dataset_params.get_model_params(bop_path, bop_dataset_name, model_type=model_type if model_type else None)
    models_dir = Path(bop_dataset_path) / 'models'
    model_paths = [x for x in models_dir.iterdir() if str(x)[-4:]=='.ply']
    model_p = {
        'obj_ids' : list(range(1,len(model_paths)+1)),
        'symmetric_obj_ids': [],
        'model_tpath': str(models_dir / 'obj_{obj_id:06d}.ply'),
        'models_info_path': None
    }
    #########################x

    assert object_model_unit in ['m', 'dm', 'cm', 'mm'], (f"Invalid object model unit: `{object_model_unit}`. "
                                                          f"Supported are 'm', 'dm', 'cm', 'mm'")
    scale = {'m': 1., 'dm': 0.1, 'cm': 0.01, 'mm': 0.001}[object_model_unit]
    if mm2m is not None:
        warnings.warn("WARNING: `mm2m` is deprecated, please use `object_model_unit='mm'` instead!")
        scale = 0.001

    if obj_ids is None:
        obj_ids = []

    obj_ids = obj_ids if obj_ids else model_p['obj_ids']

    loaded_objects = []
    # if sampling is enabled
    if sample_objects:
        loaded_ids = {}
        loaded_amount = 0
        if obj_instances_limit != -1 and len(obj_ids) * obj_instances_limit < num_of_objs_to_sample:
            raise RuntimeError(f"{bop_dataset_path}'s contains {len(obj_ids)} objects, {num_of_objs_to_sample} object "
                               f"where requested to sample with an instances limit of {obj_instances_limit}. Raise "
                               f"the limit amount or decrease the requested amount of objects.")
        while loaded_amount != num_of_objs_to_sample:
            random_id = choice(obj_ids)
            if random_id not in loaded_ids:
                loaded_ids.update({random_id: 0})
            # if there is no limit or if there is one, but it is not reached for this particular object
            if obj_instances_limit == -1 or loaded_ids[random_id] < obj_instances_limit:
                cur_obj = _BopLoader.load_mesh(random_id, model_p, bop_dataset_name, scale)
                loaded_ids[random_id] += 1
                loaded_amount += 1
                loaded_objects.append(cur_obj)
            else:
                print(f"ID {random_id} was loaded {loaded_ids[random_id]} times with limit of {obj_instances_limit}. "
                      f"Total loaded amount {loaded_amount} while {num_of_objs_to_sample} are being requested")
    else:
        for obj_id in obj_ids:
            cur_obj = _BopLoader.load_mesh(obj_id, model_p, bop_dataset_name, scale)
            loaded_objects.append(cur_obj)
    # move the origin of the object to the world origin and on top of the X-Y plane
    # makes it easier to place them later on, this does not change the `.location`
    # This is only useful if the BOP objects are not used in a pose estimation scenario.
    if move_origin_to_x_y_plane:
        for obj in loaded_objects:
            obj.move_origin_to_bottom_mean_point()

    return loaded_objects






parser = argparse.ArgumentParser()
parser.add_argument('--bop_dir', default="bop_datasets", help="Path to the bop datasets parent directory")
parser.add_argument('--cctextures_dir', default="cctextures", help="Path to downloaded cc textures")
parser.add_argument('--out_dir', default=".", help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=400, help="How many scenes with 25 images each to generate")
args = parser.parse_args()

bproc.init()

# load bop objects into the scene
#target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_dir, 'ycbv'), mm2m = True)
#target_bop_objs = load_custom_objs(bop_dataset_path = os.path.join(args.bop_dir, 'ycbv'), mm2m = True)
target_objs = load_custom_objs(str(Path(__file__).parent / 'custom_dataset'), mm2m = True)
N_target_objs = len(target_objs)

# load distractor bop objects
tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_dir, 'tless'), model_type = 'cad', mm2m = True)
ycbv_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_dir, 'ycbv'), mm2m = True)
#hb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_dir, 'hb'), mm2m = True)

# load BOP datset intrinsics - use YCBV intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_dir, 'ycbv'))

# set shading and hide objects
#for obj in (target_bop_objs + tless_dist_bop_objs + ycbv_dist_bop_objs + hb_dist_bop_objs):
for obj in (target_objs + tless_dist_bop_objs + ycbv_dist_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)


# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cctextures_dir)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([-0.2, -0.2, 0.0], [0.2, 0.2, 0.6]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
#bproc.renderer.set_render_devices(use_only_cpu=True)

for i in range(args.num_scenes):

    # Sample bop objects for a scene
    
    #sampled_target_bop_objs = list(np.random.choice(target_objs, size=20, replace=False))
    sampled_target_bop_objs = target_objs

    #sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=2, replace=False))
    #sampled_distractor_bop_objs += list(np.random.choice(ycbv_dist_bop_objs, size=2, replace=False))
    #sampled_distractor_bop_objs += list(np.random.choice(hb_dist_bop_objs, size=2, replace=False))
    sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=4, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(ycbv_dist_bop_objs, size=4, replace=False))
    

    # Randomize materials and set physics
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):        
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.1, 0.9)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)


    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs + sampled_distractor_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)

    cam_poses = 0
    while cam_poses < 25:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.65,#0.45,
                                radius_max = 1.5,#1.08,
                                elevation_min = 5,
                                elevation_max = 89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        #poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=15, replace=False))
        poi = bproc.object.compute_poi(target_objs)

        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(os.path.join(args.output_dir),
                           target_objects = sampled_target_bop_objs,
                           dataset = 'custom_dataset',
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):      
        obj.disable_rigidbody()
        obj.hide(True)





