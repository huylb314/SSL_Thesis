import argparse
import functools
import itertools
import os
import random
from copy import deepcopy

import yaml
import time
import cv2
import gym
import numpy as np
import rospy
from gym.envs.registration import register
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation as R
from tami_gym_ros.task_envs.bin_picking_sim_task import BinPickingEnv
from tami_gym_ros.utils.utils import load_meshes
from tami_rl_agent.algorithms.consac import ConSAC
from tami_rl_agent.ssl_models.fixmatch.consac_fixmatch import ConSACFixMatch
from tami_rl_agent.utils.geometry import angle_between_vectors, location_pixel, point_in_hull, sample_hull, array_pcl_to_pcl, get_points_hull
from tami_rl_agent.utils.object_detection import ChangeDetector
from tami_rl_agent.utils.visualization import (
    generate_grasp_on_height_map_image,
    rotate_image,
    rotate_pixel,
    visualize_object_detection,
)

from tami_rl_agent.utils.geometry import (
    pcl_cropping,
    point_in_hull,
    orthographic_projection,
    pcl_to_voxels,
    grid_sizes,
    pcl_to_mesh,
    array_pcl_to_pcl,
    sample_hull,
)

from tami_trainer.simple_trainer import SimpleTrainer
from tami_rl_agent.utils.replay_memory import ReplayMemory
import torch

def grasp_visualization(action, obs):
    """Generates an image for visualizing the grasp w.r.t. the current heightmap

    Args:
        action ([type]): [description]
        obs ([type]): [description]

    Returns:
        [type]: [description]
    """
    angle = action["yaw"]
    img = obs["visual"]["image"]
    heightmap = np.concatenate((obs["visual"]["loc_map"], img[:, :, 1:]), axis=2)
    grasp_img = generate_grasp_on_height_map_image(height_map=heightmap, position=action["position"], angle=angle)
    return {"grasp from height map": grasp_img}


def parse_idx_params_to_binpicking_action(action):
    """Transform array action to a bin-picking dict action according to the Bin-picking action space

    Args:
        action (np.ndarray): The array action returned from the alogirhtm

    Returns:
        [dict]: Specification of the grasp position rotation and width
    """
    params_n = action["params_scaled"]
    rot = R.from_euler("ZYX", np.array(params_n[:3][::-1])).as_euler("xyz")
    bpa = {"position": np.array(action["location"], dtype=np.float32), "rotation": np.array(rot, dtype=np.float32)}
    if len(params_n) > 3:
        bpa["width"] = np.array([params_n[3]], dtype=np.float32)
    return bpa


def action_masking(env: BinPickingEnv, obs: dict):
    """Create a mask specifying for each pixel of the observation whether it is feasible

    Args:
        env (BinPickingEnv): the bin-picking environment
        obs (dict): the observation

    Returns:
        [np.ndarray[bool]]: each value in the mask specifies if the pixel is feasible
    """
    detector = ChangeDetector(env.get_ref_pcl())
    pcl = obs["visual"]["pcl"]
    points, _, _ = detector.get_change_points_with_normals(pcl, display=False)
    locations = obs["visual"]["loc_map"].transpose([1, 2, 0])
    bin_hull_param = env.get_bin_hull(margin=0.015)
    # mask = hull_mask(locations, bin_hull)
    # return mask
    mask = np.zeros(shape=locations.shape[:2])
    for point in points:
        if not point_in_hull(point[:2], bin_hull_param):
            continue
        pixel = location_pixel(locations, point)
        mask[pixel] = 1
    return mask


def annotate_grasp_speculative(
    env_picking: BinPickingEnv,
    obs: dict,
    action: dict,
    obj_points=None,
    obj_normals=None,
    dist_threshold: float = 5e-3,
    cup_size: float = 1.5e-2,
    angle_threshold: float = 30,
    angle_mean_threshold: float = 10,
    method: str = "points",
):
    # Debug
    # env.visualize_grasp(action['location'], action['params_scaled'], width=None, scene_pcl=obs['visual']['pcl'])
    # Calculate objects pcl based on change detection if not provided in input
    if obj_points is None or obj_normals is None:
        detector = ChangeDetector(env_picking.get_ref_pcl())
        pcl = obs["visual"]["pcl"]
        points, normals, _ = detector.get_change_points_with_normals(pcl, display=False)
        # bin_hull = env_picking.get_bin_hull(margin=0.015)
        # obj_idxs = []
        # for idx, point in enumerate(points):
        #     if point_in_hull(point[:2], bin_hull):
        #         obj_idxs.append(idx)
        obj_points = points  # [obj_idxs]
        obj_normals = normals  # [obj_idxs]

    rot_euler = R.from_euler("ZYX", np.array(action["params_scaled"][:3][::-1])).as_euler("xyz")

    if not env_picking.is_parallel_gripper() and obj_normals is not None:
        # Check if location is sufficiently close to an object
        dist_point_objs = np.linalg.norm(obj_points - action["location"], axis=1)
        print(f"**** Minimal distance to pcd: {dist_point_objs.min()}")
        if dist_point_objs.min() > dist_threshold:
            return 0

        # Check if normal is aligned with gripper
        proximal_idxs = np.where(dist_point_objs < cup_size / 2.0)[0]
        print(f"proximal normals: {obj_normals[proximal_idxs]}")

        # Option1: point normal is the mean of the proximal normals
        point_normal = np.mean(obj_normals[proximal_idxs], axis=0)
        point_normal /= np.linalg.norm(point_normal)
        print(f"point normal #1: {point_normal}")
        # Option2: point normal is eigen vector with smallest eigen value of auto-product normals
        M_p = np.stack([n[:, np.newaxis] * n[:, np.newaxis].T for n in obj_normals[proximal_idxs]]).sum(axis=0)
        w, F_p = np.linalg.eig(M_p)
        smallest_ev = w.argmax()
        point_normal2 = F_p[:, smallest_ev]
        print(f"point normal #2: {point_normal2}")
        print(f"Fp:{F_p}, vals:{w}")

        rot_matrix = R.from_euler("ZYX", np.array(action["params_scaled"][:3][::-1])).as_matrix()
        orientation_matrix = np.dot(rot_matrix, R.from_euler("xyz", [-np.pi, 0, 0]).as_matrix())
        gripper_direction = orientation_matrix[:, 2]
        print(f"gripper direction: {gripper_direction}")
        angle = angle_between_vectors(point_normal, -gripper_direction)
        print(f"****  Angle between normal and gripper: {np.rad2deg(angle)}")
        if angle > np.deg2rad(angle_threshold):
            return 0

        # Evaluate if surface around the grasp point is flat using std of angels of surrounding points
        angles = [angle_between_vectors(point_normal, obj_normals[proximal_idx]) for proximal_idx in proximal_idxs]
        angle_std = np.std(angles)
        angle_mean = np.mean(angles)
        print(f"**** normals angles: {np.rad2deg(angles)}")
        print(f"**** Normal diff mean: {np.rad2deg(np.mean(angles))}")
        print(f"**** Normal diff std: {np.rad2deg(angle_std)}")
        if angle_mean > np.deg2rad(angle_mean_threshold):
            return 0

    # Check if grasp is inside the bin area and collision-free
    width = action["params_scaled"][3] if len(action["params_scaled"]) > 3 else None
    if obj_points is not None:
        return float(
            env_picking.is_feasible(
                action["location"],
                rot_euler,
                width,
                scene_pcl=np.hstack([obj_points, np.ones_like(obj_points)]),
                method=method,
                display=False,
                check_fingers=env_picking.is_parallel_gripper(),
            )
        )
    return False


def generate_speculative_grasp_samples(
    env_picking: BinPickingEnv,
    obs: dict,
    n_samples: int,
    p_pos: float = 0.5,
    ratio_strictness_level: float = 0.99,
    p_obj: float = 0.95,
    method: str = "points",
):
    max_pos = int(n_samples * p_pos)
    max_neg = n_samples - max_pos

    detector = ChangeDetector(env_picking.get_ref_pcl())
    pcl = obs["visual"]["pcl"]
    # Find object points based on change detection
    points, normals, _ = detector.get_change_points_with_normals(pcl, display=False)
    bin_hull_param = env_picking.get_bin_hull(margin=0.015)
    obj_idxs = []
    locations = obs["visual"]["loc_map"].transpose([1, 2, 0])
    objects_mask = np.zeros(shape=locations.shape[:2])
    for idx, point in enumerate(points):
        if point_in_hull(point[:2], bin_hull_param):
            obj_idxs.append(idx)
            pixel = location_pixel(locations, point)
            objects_mask[pixel] = 1
    obj_points = points[obj_idxs]
    obj_normals = normals[obj_idxs]
    samples = []
    # Pixel samping distribution biased towards object pixels w.p. p_obj
    p = p_obj / objects_mask.sum() * objects_mask + (1 - objects_mask) * (1 - p_obj) / (1 - objects_mask).sum()
    n_pos = 0
    n_neg = 0
    while n_pos + n_neg < n_samples:
        print(f"n_pos:{n_pos}, n_neg:{n_neg}", end="\r")
        idx = np.random.choice(range(np.prod(p.shape)), p=p.flatten())
        pixel = np.unravel_index(idx, p.shape)
        location = locations[pixel]
        rot = np.random.uniform(-0.785, 0.785, 3)
        # rot = R.from_euler('ZYX', rot).as_euler('xyz')
        params_scaled = rot
        if env_picking.is_parallel_gripper():
            params_scaled = np.append(params_scaled, np.random.uniform(0.04, 0.08))
        action = {"idx": idx, "params_scaled": params_scaled, "location": location}
        reward = annotate_grasp_speculative(env_picking, obs, action, obj_points, obj_normals, method=method)
        if reward:
            if n_pos >= max_pos and np.random.rand() < ratio_strictness_level:
                continue
            n_pos += 1
        else:
            if n_neg >= max_neg and np.random.rand() < ratio_strictness_level:
                continue
            n_neg += 1
        samples.append(
            {
                "obs0": obs,
                "action": action,
                "reward": reward,
                "obs1": {"visual": {"image": np.empty(0)}},
                "terminal1": True,
                "info": {},
            }
        )
    return samples


def apply_grasp_heuristic(env_picking: BinPickingEnv, obs: dict, method: str = "points"):
    pos_sample = generate_speculative_grasp_samples(
        env_picking, obs, n_samples=1, p_pos=1, ratio_strictness_level=1, p_obj=1, method=method
    )
    return pos_sample[0]["action"]


def suction_grasp_heuristic(env_picking: BinPickingEnv, obs: dict, max_trials: int = 5000, method: str = "mesh"):
    """Computes a random feasible action for the given observation

    Args:
        env (BinPickingEnv): The bin picking environment
        obs (dict): The observation
        max_trials (int, optional): maximal number of trial to find a feasible action after which an
            an arbitrary action is returned. Defaults to 5000.
        method (str, optional): The method ('point' or 'intersection') used for calculating feasibility.
            'points' check if the points of the gripper falls within the boundaries of the bin.
            'intersection' checks for mesh intersection between the gripper and . Defaults to 'points'.
        use_current_scene (bool, optional): Indicates whether to use the current observation or the
            reference (empty bin) observation for intersection. Defaults to False.

    Returns:
        [dict]: Bin picking action
    """

    # Use change detection for object detection
    detector = ChangeDetector(env_picking.get_ref_pcl())
    pcl = obs["visual"]["pcl"]
    points, normals, _ = detector.get_change_points_with_normals(pcl, display=False)
    idxs = np.arange(len(points))
    np.random.shuffle(idxs)
    bin_hull = env_picking.get_bin_hull(margin=0.05)
    locations = obs["visual"]["loc_map"].transpose([1, 2, 0])
    objects_mask = np.zeros(shape=locations.shape[:2])
    for point in points:
        if point[2] > 0.25:
            continue
        pixel = location_pixel(locations, point)
        objects_mask[pixel] = 1
    objects_mask = binary_erosion(objects_mask, structure=np.ones((5, 5)))
    objects_mask = binary_erosion(objects_mask, structure=np.ones((3, 3)))
    visualize_object_detection(obs["visual"]["image"][1:], objects_mask)
    obj_pixels = list(zip(*np.where(objects_mask)))
    random.shuffle(obj_pixels)
    # Define the action params domains
    # the deepening distance beyond the grasp point in the orientation direction
    offsets = [0.03]
    # the angle around the z-axis of the grasp orientation
    angles = np.linspace(-np.pi, np.pi, 20)
    # Define the grasp orientation z-axis (the gripper pointing direction). To restrict the orientation to
    # only point downward and restricting the picth and roll absolute value to be less than 45 degrees the
    # z component is set to -1 and the x, y components between -1 and 1.
    # the x component of the orientation z-axis (the gripper pointing direction)
    z_xs = [0]  # np.linspace(-0.01, 0.01, 3)
    # the y component of the orientation z-axis (the gripper pointing direction)
    z_ys = [0]  # np.linspace(-0.01, 0.01, 3)
    # The z component
    z_zs = [-1]
    grasp_params = list(itertools.product(z_xs, z_ys, z_zs, offsets, angles))

    grasp_point = sample_hull(bin_hull)
    rot = np.zeros(3)
    if len(points) == 0:
        pixel = location_pixel(locations, grasp_point)
        idx = np.ravel_multi_index(pixel, locations.shape[:2])
        return dict(idx=idx, params_scaled=np.array([0, 0, 0]))
    # Sort the parameters from simple (smallest pitch and roll) to less simple

    def energy(params):
        return sum(abs(np.asarray(params[:2])))

    grasp_params = sorted(grasp_params, key=energy)

    found = False
    trial = 0
    idx_point = 0
    while not found and trial < max_trials and idx_point < len(obj_pixels):
        idx_rand = idxs[idx_point]
        pixel = obj_pixels[idx_point]
        grasp_point = locations[pixel]
        idx_point += 1
        normal_normalized = normals[idx_rand] / np.linalg.norm(normals[idx_rand])
        #  Skip if the implied gripper orientation would be too blunt
        if normal_normalized[2] < 0 or np.abs(normal_normalized[2]) < np.linalg.norm(normal_normalized[:2]):
            continue
        # position = points[idx_rand]
        # Skip if position falls out of bin
        if not point_in_hull(grasp_point.copy()[:2], bin_hull):
            continue

        for z_x, z_y, z_z, offset, angle in grasp_params:
            trial += 1
            if trial % 100 == 0:
                print(f"Trial {trial} for point {idx_point}")
            # Forming the rotation matrix
            z = np.array([z_x, z_y, z_z])
            z /= np.linalg.norm(z)
            # Use an arbitrary y orthogonal to z
            y = np.array([0, 1, -z[1] / z[2]])
            y /= np.linalg.norm(y)
            # grasp_point = position - \
            #     offset * \
            #     normal_normalized
            O_gripper = np.array([np.cross(z, y), y, z]).T
            # Make sure the orientation is in right-hand side system
            if np.linalg.det(O_gripper) < 0:
                O_gripper[:3, 0] *= -1
            # Rotate the orientation around the z-axis
            O_gripper = np.dot(R.from_rotvec(angle * z, degrees=False).as_matrix(), O_gripper)
            Rot = np.dot(O_gripper, np.linalg.inv(R.from_euler("xyz", [-np.pi, 0, 0]).as_matrix()))
            # Make sure the resulting orientation is kinematically feasible (rotation of EE joint is within limits)
            yaw = R.from_matrix(Rot).as_euler("ZYX")[0]
            if yaw > (np.pi / 4.0) or yaw < (-np.pi / 4.0):
                continue
            rot = R.from_matrix(Rot).as_euler("ZYX")[::-1]
            # Check that the resulting grasp is collision-free and bounded to bin
            feasible = env_picking.is_feasible(
                grasp_point,
                rot,
                None,
                scene_pcl=np.hstack([points, np.ones_like(points)]),
                method=method,
                check_fingers=False,
                display=False,
            )
            if feasible:
                found = True
                break
    # Transform grasp to a corresponding action that would be returned from the algorithm
    print(f"****  GRASP POINT:{grasp_point}, rot:{rot}")
    # pixel = location_pixel(locations, grasp_point)
    idx = np.ravel_multi_index(pixel, locations.shape[:2])
    return dict(idx=idx, params_scaled=rot)


def grasp_heuristic(env_picking: BinPickingEnv, obs: dict, max_trials: int = 5000, method: str = "points"):
    """Computes a random feasible action for the given observation

    Args:
        env (BinPickingEnv): The bin picking environment
        obs (dict): The observation
        max_trials (int, optional): maximla number of trial to find a feasible action after which an
            an arbitraty action is returned. Defaults to 5000.
        method (str, optional): The method ('point' or 'intersection') used for calculating feasiblity.
            'points' check if the points of the gripper falls within the boundaries of the bin.
            'intersection' checks for mesh intersection beteen the gripper and . Defaults to 'points'.
        use_current_scene (bool, optional): Indicates whether to use the current observation or the
            reference (empty bin) observation for intersection. Defaults to False.

    Returns:
        [dict]: Bin picking action
    """

    # Use change detection for object detection
    detector = ChangeDetector(env_picking.get_ref_pcl())
    pcl = obs["visual"]["pcl"]
    points, normals, _ = detector.get_change_points_with_normals(pcl, display=True)
    idxs = np.arange(len(points))
    np.random.shuffle(idxs)
    bin_hull_param = env_picking.get_bin_hull(margin=0.05)

    # Define the action params domains
    # the deepening distance beyond the grasp point in the orientation direction
    offsets = [0.03]
    # the angle around the z-axis of the grasp orientation
    angles = np.linspace(-np.pi, np.pi, 10)
    # the distance between the fingers
    widths = np.linspace(0.08, 0.03, 3)
    # Define the grasp orientation z-axis (the gripper pointing direction). To restrict the orientation to
    # only point downward and restricting the picth and roll absolute value to be less than 45 degrees the
    # z component is set to -1 and the x, y components between -1 and 1.
    # the x component of the orientation z-axis (the gripper pointing direction)
    z_xs = np.linspace(-1, 1, 5)
    # the y component of the orientation z-axis (the gripper pointing direction)
    z_ys = np.linspace(-1, 1, 5)
    # The z component
    z_zs = [-1]
    grasp_params = list(itertools.product(z_xs, z_ys, z_zs, offsets, angles, widths))

    locations = obs["visual"]["loc_map"].transpose([1, 2, 0])
    grasp_point = sample_hull(bin_hull_param)
    rot = np.zeros(3)
    gripper_width = 0.08
    if len(points) == 0:
        pixel = location_pixel(locations, grasp_point)
        idx = np.ravel_multi_index(pixel, locations.shape[:2])
        return {"idx": idx, "params_scaled": np.array([0, 0, 0, 0.08])}
    # Sort the parameters from simple (smallest pitch and roll) to less simple

    def energy(params):
        return sum(abs(np.asarray(params[:2])))

    grasp_params = sorted(grasp_params, key=energy)

    found = False
    trial = 0
    idx_point = 0
    while not found and trial < max_trials and idx_point < len(idxs):
        idx_rand = idxs[idx_point]
        idx_point += 1
        normal_normalized = normals[idx_rand] / np.linalg.norm(normals[idx_rand])
        #  Skip if the implied gripper orientation would be too blunt
        if normal_normalized[2] < 0 or np.abs(normal_normalized[2]) < np.linalg.norm(normal_normalized[:2]):
            continue
        position = points[idx_rand]
        # Skip if position falls out of bin
        if not point_in_hull(position.copy()[:2], bin_hull_param):
            continue

        for z_x, z_y, z_z, offset, angle, gripper_width in grasp_params:
            trial += 1
            if trial % 100 == 0:
                print(f"Trial {trial} for point {idx_point}")
            # Forming the rotation matrix
            z = np.array([z_x, z_y, z_z])
            z /= np.linalg.norm(z)
            # Use an arbitrary y orthogonal to z
            y = np.array([0, 1, -z[1] / z[2]])
            y /= np.linalg.norm(y)
            grasp_point = position - offset * normal_normalized
            O_gripper = np.array([np.cross(z, y), y, z]).T
            # Make sure the orientation is in right-hand side system
            if np.linalg.det(O_gripper) < 0:
                O_gripper[:3, 0] *= -1
            # Rotate the orientation around the z-axis
            O_gripper = np.dot(R.from_rotvec(angle * z, degrees=False).as_matrix(), O_gripper)
            Rot = np.dot(O_gripper, np.linalg.inv(R.from_euler("xyz", [-np.pi, 0, 0]).as_matrix()))
            # Make sure the resulting orientation is kinematically feasible (rotation of EE joint is within limits)
            yaw = R.from_matrix(Rot).as_euler("ZYX")[0]
            if yaw > np.pi or yaw < -np.pi / 2.0:
                continue
            rot = R.from_matrix(Rot).as_euler("ZYX")[::-1]
            # Check that the resulting grasp is collision-free and bounded to bin
            feasible = env_picking.is_feasible(
                grasp_point,
                rot,
                gripper_width,
                scene_pcl=np.hstack([points, np.ones_like(points)]),
                method=method,
                check_fingers=False,
                display=True,
            )
            if feasible:
                found = True
                break
    # Transform grasp to a corresponding action that would be returned from the algorithm
    pixel = location_pixel(locations, grasp_point)
    idx = np.ravel_multi_index(pixel, locations.shape[:2])
    return dict(idx=idx, params_scaled=np.append(rot, gripper_width))


def random_transformation(batch, angle_limit=np.rad2deg(10), pixel_limit=10, p_trans=0.1):
    batch_trans = deepcopy(batch)
    for i in range(len(batch.reward)):
        if np.random.rand() > p_trans:
            continue
        img = batch_trans.obs0[i].copy()
        img_cv2 = img.transpose(1, 2, 0)
        img_size = np.array(img_cv2.shape[:2])
        # Translate
        pixel = np.array(np.unravel_index(int(batch.action["idx"][i]), img_cv2.shape[:2]))
        img = img_cv2.copy()
        # img[pixel[0], pixel[1], 1:] = [0, 0, 255]
        pixel_translation = np.random.randint(
            np.maximum(-pixel_limit, -pixel), np.minimum(pixel_limit, img_size - 1 - pixel)
        )
        img_translated = np.zeros(np.append(img_size + 2 * pixel_limit, 4))
        orig = pixel_limit + pixel_translation
        img_translated[orig[0] : orig[0] + img_size[0], orig[1] : orig[1] + img_size[1]] = img_cv2
        img_translated = img_translated[
            pixel_limit : pixel_limit + img_size[0], pixel_limit : pixel_limit + img_size[1]
        ]
        pixel_translated = pixel + pixel_translation
        # img_translated[pixel_translated[0], pixel_translated[1], 1:] = [0, 0, 255]
        # Rotate
        angle = np.random.uniform(-angle_limit, angle_limit)
        img_transformed = rotate_image(img_translated, angle)
        pixel_tranformed = rotate_pixel(pixel_translated, angle, img_size)
        params_scaled = batch.action["params_scaled"][i].copy()
        action_params = batch.action["params"][i].copy()
        euler = params_scaled[:3]
        scale = np.array([np.pi / 4, np.pi / 4, np.pi / 4])  # params_scaled[:3] / params[:3]
        rot = R.from_euler("xyz", euler).as_matrix()
        rot_aug = R.from_euler("xyz", [0, 0, angle]).as_matrix()
        rot_aug = np.dot(rot_aug, rot)
        euler_aug = R.from_matrix(rot_aug).as_euler("xyz")
        batch_trans.action["idx"][i] = np.ravel_multi_index(pixel_tranformed, img_size)
        params_scaled[:3] = euler_aug
        action_params[:3] = params_scaled[:3] / scale
        batch_trans.action["params_scaled"][i] = params_scaled
        batch_trans.action["params"][i] = action_params
        batch_trans.obs0[i] = img_transformed.transpose(2, 0, 1)
    return batch_trans


def run_trainer(
    env_picking: gym.Env,
    agent_consac: ConSAC,
    steps: int,
    model_path: str,
    memory_path: str,
    debug_output_path: str,
    without_online_sample: bool,
):
    params = {"gamma": agent_consac._discount, "n_rnd_ini": 0, "max_episode_steps": 25, "n_updates": 100}
    trainer = SimpleTrainer(
        algorithm=agent_consac,
        env_id="BinPicking-v0",
        exp_id="convsac",
        parameters=params,
        use_clearml=False,
        action_parser=parse_idx_params_to_binpicking_action,
        eval_rate=1, # set 1 for every step evaluation (default 10)
        debug_output_path=debug_output_path,
        without_online_sample=without_online_sample,
    )
    trainer.train(env=env_picking, steps=steps, model_path=model_path, memory_path=memory_path)

# def display_imgs(input_image, heatmap, title="image"):
#     rgb_title = f"{title}_rgb"
#     depth_title = f"{title}_depth"
#     normalized_title = f"{title}_normalized"
#     heatmap_title = f"{title}_heatmap"

#     rgb = input_image[:, :, 1:4]
#     rgb = rgb.astype(np.uint8)
#     depth = input_image[:, :, :1]
#     depth = (255 * (depth - depth.min()) / (depth.max() - depth.min())).astype(np.uint8)

#     normalized = input_image[:, :, 4:]
#     normalized = (255 * (normalized - normalized.min()) / (normalized.max() - normalized.min())).astype(np.uint8)

#     cv2.namedWindow(rgb_title, cv2.WINDOW_NORMAL)
#     cv2.namedWindow(depth_title, cv2.WINDOW_NORMAL)
#     # cv2.namedWindow(normalized_title, cv2.WINDOW_NORMAL)
#     # cv2.namedWindow(heatmap_title, cv2.WINDOW_NORMAL)
#     cv2.imshow(rgb_title, rgb)
#     cv2.imshow(depth_title, depth)
#     # cv2.imshow(normalized_title, normalized)
#     # cv2.imshow(heatmap_title, heatmap)

#     cv2.resizeWindow(rgb_title, 800, 800)
#     cv2.resizeWindow(depth_title, 800, 800)
#     # cv2.resizeWindow(normalized_title, 800, 800)
#     # cv2.resizeWindow(heatmap_title, 800, 800)
#     # here it should be the pause
#     k = cv2.waitKey(0)
#     if k == 27:  # wait for ESC key to exit
#         cv2.destroyAllWindows()
        
def display_imgs(input_image, title="image"):
    rgb_title = f"{title}"
    
    cv2.namedWindow(rgb_title, cv2.WINDOW_NORMAL)
    cv2.imshow(rgb_title, input_image)

    cv2.resizeWindow(rgb_title, 800, 800)
    # here it should be the pause
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        
# def orthographic_projection(pcl, orig, size, resolution, flipud=False):
#     """
#     Project a point cloud onto the XY-hyperplane clipped to a region of interest bounding box
#     Args:
#         flipud: flip the output so that positive y direction would go upwards
#         pcl: (N, 6) ndarray
#         pixel_density: the number of pixels per a unit of measurement (int)
#         bbox_center: the center of region
#         bbox_half_sizes: 3d array representing the distance from center of the region in each dimension
#         take_max_z: bool which specified whether to take the max z value of all points that fall in a pixel or take one
#         of the z values arbitrarily (slightly faster)

#     Returns:
#         an image with 6 channels (x, y, z, r, g, b)
#     """
#     n_voxels = size / resolution
#     n_voxels = (np.ceil(n_voxels / 8) * 8).astype(np.uint16)

#     lb = orig
#     ub = orig + size
#     reduced_pcl = pcl
#     reduced_pcl = pcl[np.all(pcl[:, :3] >= lb, axis=1) & np.all(pcl[:, :3] <= ub, axis=1), :]
#     xy_image = np.zeros((n_voxels[1], n_voxels[0], pcl.shape[1]))
#     if pcl.shape[1] > 8:
#         xy_image[:, :, 8] = 1
#     xy_image[:, :, 2] = lb[2]
#     z_image = lb[2] * np.ones((n_voxels[1], n_voxels[0], n_voxels[2]))
#     xx, yy = np.meshgrid(range(n_voxels[0]), range(n_voxels[1]), indexing="xy")
#     xy_image[:, :, 0] = lb[0] + np.array(0.5 + xx, dtype=np.float32) * resolution
#     xy_image[:, :, 1] = lb[1] + np.array(0.5 + yy, dtype=np.float32) * resolution
#     pixels = np.array(((reduced_pcl[:, :3] - lb[:3]) // resolution), dtype=np.int32)
#     xy_image[pixels[:, 1], pixels[:, 0], 2:] = reduced_pcl[:, 2:]
#     z_image[pixels[:, 1], pixels[:, 0], pixels[:, 2]] = reduced_pcl[:, 2]
#     xy_image[:, :, 2] = np.max(z_image, axis=2)
#     if flipud:
#         xy_image = np.flipud(xy_image)
#     return xy_image

def calculate_bbox(pick_area_corners):
    """
    A bounding box expressed as a tuple of center and half sizes, covering the bin area with added margin
    Args:
        margin: scalar, the size of the added margin in meters

    Returns: a center (x, y, z) and half sizes (s_x, s_y, s_z) of the box in each axis

    """
    bin_corners = np.array(pick_area_corners)
    #  = bin_corners.mean(axis=0)
    # diag = np.max(np.vstack([np.abs(p - center) for p in points]), axis=0)
    # diag = np.max([np.linalg.norm(p[:2] - center[:2]) for p in points])
    # diag = np.full(2, diag)
    bin_max = bin_corners.max(axis=0)
    bin_min = bin_corners.min(axis=0)
    bbox_center = (bin_max + bin_min) / 2.0
    bbox_hsize = (bin_max - bin_min) / 2.0 + 0.05
    bbox_hsize[2] += 0.05
    # center[2] += self.task_params['bin_height'] / 2
    return bbox_center, bbox_hsize
    # , np.append(diag[:2], (points[:, 2].max() - points[:, 2].min()) / 2.) * 1.1

# def load_bin(bin_config, margin=0):
#     with open(bin_config, "r") as f:
#         params = yaml.safe_load(f)
#         bin_corners = np.array(params["pick_area_corners"])
#         bin_hull = get_points_hull(bin_corners, margin=margin)
#         bin_max = bin_corners.max(axis=0)
#         bin_min = bin_corners.min(axis=0)
#         bbox_center = (bin_max + bin_min) / 2.0
#         bbox_hsize = (bin_max - bin_min) / 2.0 + 0.05
#         bbox_hsize[2] += 0.05  # bigger margin for the z direction
        
#         return bin_hull, bbox_center, bbox_hsize

# def orthographic_projection(pcl, orig, size, resolution, flipud=False):
#     """
#     Project a point cloud onto the XY-hyperplane clipped to a region of interest bounding box
#     Args:
#         flipud: flip the output so that positive y direction would go upwards
#         pcl: (N, 6) ndarray
#         pixel_density: the number of pixels per a unit of measurement (int)
#         bbox_center: the center of region
#         bbox_half_sizes: 3d array representing the distance from center of the region in each dimension
#         take_max_z: bool which specified whether to take the max z value of all points that fall in a pixel or take one
#         of the z values arbitrarily (slightly faster)

#     Returns:
#         an image with 6 channels (x, y, z, r, g, b)
#     """
#     n_voxels = size / resolution
#     n_voxels = (np.ceil(n_voxels / 8) * 8).astype(np.uint16)

#     lb = orig
#     ub = orig + size
#     reduced_pcl = pcl
#     reduced_pcl = pcl[np.all(pcl[:, :3] >= lb, axis=1) & np.all(pcl[:, :3] <= ub, axis=1), :]
#     xy_image = np.zeros((n_voxels[1], n_voxels[0], pcl.shape[1]))
#     if pcl.shape[1] > 8:
#         xy_image[:, :, 8] = 1
#     xy_image[:, :, 2] = lb[2]
#     z_image = lb[2] * np.ones((n_voxels[1], n_voxels[0], n_voxels[2]))
#     xx, yy = np.meshgrid(range(n_voxels[0]), range(n_voxels[1]), indexing="xy")
#     xy_image[:, :, 0] = lb[0] + np.array(0.5 + xx, dtype=np.float32) * resolution
#     xy_image[:, :, 1] = lb[1] + np.array(0.5 + yy, dtype=np.float32) * resolution
#     pixels = np.array(((reduced_pcl[:, :3] - lb[:3]) // resolution), dtype=np.int32)
#     xy_image[pixels[:, 1], pixels[:, 0], 2:] = reduced_pcl[:, 2:]
#     z_image[pixels[:, 1], pixels[:, 0], pixels[:, 2]] = reduced_pcl[:, 2]
#     xy_image[:, :, 2] = np.max(z_image, axis=2)
#     if flipud:
#         xy_image = np.flipud(xy_image)
#     return xy_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tossingbot Trainer")
    parser.add_argument(
        "-r",
        "--robot",
        dest="robot",
        type=str,
        default="panda_vacuum",
        help="The robot used for bin picking",
    )
    parser.add_argument("-e", "--env_id", dest="env_id", type=str, default="BinPicking-v1", help="The task env")
    parser.add_argument("--path", dest="path", type=str, default="", help="The path of the RPC data dir")
    parser.add_argument("--log_dir", dest="log_dir", type=str, default="real_e32", help="The path to save the memory")
    parser.add_argument(
        "-c",
        "--conf",
        dest="config_file",
        type=str,
        default="bin_picking_large_bin_tue_suc",
        # default='bin_picking_2bins_real_suction',
        help="The name of the env config file",
    )
    parser.add_argument(
        "--mem_path",
        dest="memory_path",
        type=str,
        default="suction_pick_small_object_2",
        help="The path to save the memory",
    )
    parser.add_argument(
        "--init_mem_paths",
        dest="init_memory_paths",
        type=str,
        default="",
        help="The paths to load init memory separated by :",
    )
    parser.add_argument(
        "--encoder",
        dest="encoder",
        type=str,
        default="Resnet43_8s",
        help="The pixel encoder architecture used in ConSAC",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default="adaptive_continual_learning_models_2",
        help="The path to save the memory",
    )
    parser.add_argument(
        "--init_model_path", dest="init_model_path", type=str, default="", help="The path to save the memory"
    )
    parser.add_argument(
        "--feas_model_path",
        dest="feasibility_model_path",
        type=str,
        default="Collision/panda_hfa_suction_gripper/2bins_real_rand_binary",
        help="The path to feasibility model",
    )
    parser.add_argument(
        "--steps", 
        dest="total_steps", 
        type=int, 
        default=1000, 
        help="The number of training steps to run"
    )
    parser.add_argument(
        "--add_offline_data",
        dest="add_offline_data",
        action="store_true",
        help="If to load offline data to replay buffer",
    )
    parser.add_argument(
        "--config_model_path",
        dest="config_model_path",
        type=str,
        default="~/amira_experiments",
        help="The path for configs and model",
    )
    parser.add_argument(
        "--data_path", dest="data_path", type=str, default="new_unseen_data", help="The path to save the memory"
    )
    parser.add_argument(
        "--bc",
        "--bin_conf",
        dest="bin_config_file",
        type=str,
        default="2bins_real_tue",
        help="The path to config and model",
    )
    parser.add_argument(
        "--probabilities",
        dest="probabilities",
        action="store_true",
        help="Whether to train or evaluate algorithm probabilistically. "
        + "If True, the algorithm samples grasp points according to their probabilities "
        + "instead of deterministically choosing the point with highest probability",
    )
    parser.add_argument("--visualize", dest="visualize", action="store_true", help="If to visualize evaluation")
    parser.add_argument(
        "--without_online_sample",
        dest="without_online_sample",
        action="store_true",
        help="Don't take online sample while training",
    )
    parser.add_argument(
        "--debug_output_path",
        dest="debug_output_path",
        type=str,
        default="",
        help="Save results from online sample for debug.",
    )
    parser.add_argument(
        "--config_path", dest="config_path", type=str, default="~/amira_experiments", help="The path for configs files"
    )
    parser.add_argument(
        "--gc",
        "--gripper_conf",
        dest="gripper_config_file",
        type=str,
        default="panda_hfa_suction_gripper",
        help="The name of the env config file",
    )
    parser.add_argument(
        "--model_finetune_path",
        dest="model_finetune_path",
        type=str,
        default=None,
        help="The path to load the starting model",
    )

    parser.add_argument(
        "--train_epochs", dest="train_epochs", type=int, default=50, help="The number of training epochs to run"
    )
    parser.add_argument(
        "--vis_freq", dest="visualization_freq", type=int, default=10, help="The number of EPOCHS to save visualization"
    )
    parser.add_argument(
        "--freq_epoch", dest="freq_epoch", type=int, default=100, help="The number of EPOCHS to save state"
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=1, help="The batch size to train the model"
    )
    parser.add_argument(
        "--w_offline", dest="w_offline", type=float, default=1.0, help="The loss coefficient for offline training"
    )
    parser.add_argument(
        "--w_online", dest="w_online", type=float, default=1.0, help="The loss coefficient for online training"
    )
    parser.add_argument(
        "--w_pseudo", dest="w_pseudo", type=float, default=1.0, help="The loss coefficient for pseudo data training"
    )
    parser.add_argument(
        "--ssl_confidence", dest="ssl_confidence", type=float, default=0.8, help="The confidence threshold for pseudo label"
    )
    parser.add_argument(
        "--recalc_reward",
        action="store_true",
        dest="recalc_reward",
        help="reclac the reward even if already calculated.",
    )
    parser.add_argument('--eval_only',
                        action="store_true",
                        default=False,
                        help="only evaluate the model.")
    parser.add_argument('--train_only',
                        action="store_true",
                        default=False,
                        help="only train the model.")
    parser.add_argument('--normalize_rewards',
                        action="store_true",
                        default=False,
                        help="normalize offline data rewards.")
    parser.add_argument('--ssl_method',
                        default='fixmatch',
                        help="select ssl method (default: fixmatch)",
                        choices=['fixmatch','flexmatch', 'freematch', 'freematch_ent', 'softmatch'])
    parser.add_argument(
        "--note", dest="note", type=str, default="", help="note for tensorboard log & visualization"
    )
    parser.add_argument('--seed', dest="seed", type=int, default=-1, help="set random seed for reproducible")
    parser.add_argument(
        "--limit", dest="limit", type=int, default=None, help="online data memory buffer limit"
    )
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for evaluating model')
    parser.add_argument('--lr', dest="learning_rate", type=float, default=1e-4, help='learning rate')
        
    args = parser.parse_args()
    

    if args.robot == "panda":
        act_limit = np.array([[-0.785, 0.785], [-0.785, 0.785], [-1.57, 3.14], [0.03, 0.08]])
    else:
        act_limit = np.array([[-0.785, 0.785], [-0.785, 0.785], [-0.785, 0.785]])
    feasibility_act_limit = np.array([[-0.785, 0.785], [-0.785, 0.785], [-0.785, 0.785]])

    # Haifa bin's HM size
    # hm_size = np.array([0.5, 0.7, 0.7])
    # hm_resolution = 0.0022
    hm_resolution = 0.0035 # Voxel resolution

    # Create agent
    net_config = dict(
        in_channels=7,
        embedding_size=64,
        encoder=args.encoder,
        act_limit=act_limit,
        apply_normalize=True,
        pi_activation="tanh",
        q_activation="sigmoid",  # "linear" for non-ssl
    )

    feasibility_net_config = dict(
        in_channels=7,
        embedding_size=64,
        encoder=args.encoder,
        act_limit=feasibility_act_limit,
        apply_normalize=True,
        pi_activation="tanh",
        q_activation="sigmoid",
    )

    params = dict(
        net_cfg=net_config,
        feasibility_net_cfg=feasibility_net_config,
        act_limit=act_limit,
        epsilon_start=1,
        epsilon_end=0.1,
        epsilon_decay=1000,
        discount=0.0,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        critic_loss="BCE",  # "MSE" for non-ssl
        use_augmentation=True,
        sampler="random",
        # sampler_params=dict(pos_size=int(args.batch_size / 2)),  # use this if sampler="pos_neg"
        limit=args.limit,
        visualize=args.visualize,
        visualization_freq=args.visualization_freq,
        # for ssl learning (pseudo-labeling)
        w_offline=args.w_offline,
        w_online=args.w_online,
        w_pseudo=args.w_pseudo,
        ssl_confidence=args.ssl_confidence,
        ssl_method=args.ssl_method,
        ema_m=args.ema_m,
        note=args.note
    )
    model_path = f"{args.config_model_path}/BinPicking-v1/{args.robot}/{args.gripper_config_file}/{args.model_path}"
    model_path = os.path.expanduser(model_path)
    
    agent = ConSACFixMatch(params)

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    bin_config_file = f'{args.bin_config_file}.yaml'
    # bin_hull, bbox_center, bbox_hsize = load_bin(f'{os.path.join(args.config_path, bin_config_file)}')
    # load mesh
    gripper_config_file = f'{args.gripper_config_file}.yaml'
    gripper_meshes = load_meshes(f'{os.path.join(args.config_path, gripper_config_file)}')
    # hm_size = 2 * bbox_hsize
    # calc
    hm_resolution = 0.0035

    ref_pcl_path = os.path.expanduser("~/amira_experiments/datasets/2bins_real_tue/near_wall_realsense/")
    # ref_pcl = cv2.imread(os.path.join(ref_pcl_path, "ref_pcl_tue_lab__.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    ref_pcl = cv2.imread(os.path.join(ref_pcl_path, "ref_pcl_0707setup_test.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # o3d.visualization.draw_geometries([array_pcl_to_pcl(ref_pcl)])
    pcd = array_pcl_to_pcl(ref_pcl)
    pcd.estimate_normals()
    pcd.normalize_normals()
    pcl = np.hstack((ref_pcl, pcd.normals))
    
    with open(f'/home/leh2rng/panda_ws/src/tami_gym_ros/config/bin_picking_large_bin_tue_suc.yaml', "r") as f:
        params = yaml.safe_load(f)
        bin_corners = np.array(params["pick_area_corners"])
    
        #  = bin_corners.mean(axis=0)
        # diag = np.max(np.vstack([np.abs(p - center) for p in points]), axis=0)
        # diag = np.max([np.linalg.norm(p[:2] - center[:2]) for p in points])
        # diag = np.full(2, diag)
        bin_max = bin_corners.max(axis=0)
        bin_min = bin_corners.min(axis=0)
        bbox_center = (bin_max + bin_min) / 2.0
        bbox_hsize = (bin_max - bin_min) / 2.0 + 0.05
        # bbox_hsize[2] += 0.05
        # center[2] += self.task_params['bin_height'] / 2

        hm_featured = orthographic_projection(
            pcl,
            pixel_density=params["pixel_density"],
            bbox_center=bbox_center,
            bbox_half_sizes=bbox_hsize,
        )
        hm_featured[:, :, -1] = binary_erosion(hm_featured[:, :, -1], structure=np.ones((3, 3)), iterations=2)
        
        obs_ref = hm_featured[:, :, 2:9].transpose(2, 0, 1).copy()
        obs_ref[1:4] = obs_ref[1:4] / 255.
    
    # train_replay_memory = ReplayMemory(capacity=int(1e6), default_sampler="random")
    # train_replay_memory.load(os.path.expanduser(f"~/amira_experiments/memories/{args.memory_path}"))
    # train_replay_memory.mark_as_synced()    
    
    eval_replay_memory = ReplayMemory(capacity=int(1e6), default_sampler="random")
    eval_replay_memory.load(os.path.expanduser(f"~/amira_experiments/memories/{args.memory_path}_evaluation/"))
    eval_replay_memory.mark_as_synced()
    
    new_replay_memory = ReplayMemory(capacity=int(1e6), default_sampler="random")
    new_replay_memory_path = os.path.expanduser(f"~/amira_experiments/memories/{args.memory_path}_new/")
    # new_replay_memory.load(new_replay_memory_path)
    # new_replay_memory.mark_as_synced()
    
    
    # mask = cv2.cvtColor(rgb_height_map, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    threshold_encoded = 0.0005  # Adjust the threshold based on your requirements
    threshold_raw = 0.005  # Adjust the threshold based on your requirements
    obs_ref_shape = obs_ref.shape
    batch_size = 32
    
    path_save = "/home/leh2rng/panda_ws/src/eval_new_0707_test"
    
    for idx in range(len(eval_replay_memory)):
        # online_obs, online_action, online_reward, online_obs_next, online_done, _ = train_replay_memory.sample(batch_size=batch_size)
        
        obs, action, reward, obs_next, done, info = eval_replay_memory.memory[idx]
        obs_rgb = np.transpose(obs[1:4, :, :], (1, 2, 0))
        obs_raw = deepcopy(obs)
        obs_raw[1:4] = obs_raw[1:4] / 255.
        obs_raw = np.transpose(obs_raw, (1, 2, 0))
        obs_ref_t = np.transpose(obs_ref, (1, 2, 0))
        
        # % Perform background subtraction to get foreground mask
        # foregroundMaskColor = ~(sum(abs(inputColor-backgroundColor) < 0.3,3) == 3);
        # foregroundMaskDepth = backgroundDepth ~= 0 & abs(inputDepth-backgroundDepth) > 0.02;
        # foregroundMask = (foregroundMaskColor | foregroundMaskDepth);
        
        inputColor = obs_raw[:, :, 1:4]
        backgroundColor = obs_ref_t[:, :, 1:4]
        inputDepth = obs_raw[:, :, :1]
        backgroundDepth = obs_ref_t[:, :, :1]
        # # foregroundMaskColor = (np.sum(np.abs(inputColor - backgroundColor) < 0.015, axis=2) == 3)
        # foregroundMaskColor = (np.sum(np.abs(inputColor - backgroundColor) < 0.03, axis=2) == 3)
        # foregroundMaskColor = np.expand_dims(foregroundMaskColor, axis=-1)
        # # foregroundMaskDepth = (backgroundDepth != 0) & (np.abs(inputDepth - backgroundDepth) > 0.1)
        # foregroundMaskDepth = (np.abs(inputDepth - backgroundDepth) < 0.0005)
        # foregroundMask = foregroundMaskColor & foregroundMaskDepth
        foregroundMaskColor = (np.sum(np.abs(inputColor - backgroundColor) < 0.1, axis=2) == 3)
        foregroundMaskColor = np.expand_dims(foregroundMaskColor, axis=2)
        
        foregroundMaskDepth = (np.abs(inputDepth - backgroundDepth) < 0.02)
        foregroundMask = np.logical_or(foregroundMaskColor, foregroundMaskDepth)


        # mask_raw = np.zeros_like(foregroundMaskDepth, dtype=np.uint8)
        # mask_raw[foregroundMaskColor] = 255
        # mask_raw = cv2.applyColorMap(mask_raw, cv2.COLORMAP_JET)
        # mask_raw = cv2.addWeighted(mask_raw, 0.7, obs_rgb.astype(np.uint8), 0.3, 0)
        # path_save_raw = os.path.join(path_save, "raw", str("t_raw_new"))
        # path_save_raw_name = os.path.join(path_save_raw, f"{time.time()}_foregroundMaskColor.png")
        
        # if not os.path.exists(path_save_raw):
        #     os.makedirs(path_save_raw)
        
        # cv2.imwrite(path_save_raw_name, mask_raw)
        
        # mask_raw = np.zeros_like(foregroundMaskDepth, dtype=np.uint8)
        # mask_raw[foregroundMaskDepth] = 255
        # mask_raw = cv2.applyColorMap(mask_raw, cv2.COLORMAP_JET)
        # mask_raw = cv2.addWeighted(mask_raw, 0.7, obs_rgb.astype(np.uint8), 0.3, 0)
        # path_save_raw = os.path.join(path_save, "raw", str("t_raw_new"))
        # path_save_raw_name = os.path.join(path_save_raw, f"{time.time()}_foregroundMaskDepth.png")
        
        # if not os.path.exists(path_save_raw):
        #     os.makedirs(path_save_raw)
        
        # cv2.imwrite(path_save_raw_name, mask_raw)
        
        mask_raw = np.zeros_like(foregroundMaskDepth, dtype=np.uint8)
        mask_raw[foregroundMask] = 255
        mask_raw = cv2.applyColorMap(mask_raw, cv2.COLORMAP_JET)
        mask_raw = cv2.addWeighted(mask_raw, 0.7, obs_rgb.astype(np.uint8), 0.3, 0)
        path_save_raw = os.path.join(path_save, "raw", str("t_raw_new"))
        path_save_raw_name = os.path.join(path_save_raw, f"{time.time()}_foregroundMask.png")        
        
        if not os.path.exists(path_save_raw):
            os.makedirs(path_save_raw)
        
        cv2.imwrite(path_save_raw_name, mask_raw)
        
        # create mask_idx matrix
        mask_idxs = foregroundMask.astype(np.float64)
        mask_idxs = mask_idxs.flatten()
        mask_idxs[action["idx"]] = 1.0
        
        # reward matrix
        mask_rewards = np.ones_like(foregroundMask) * -1
        mask_rewards[foregroundMask] = 0
        mask_rewards = mask_rewards.flatten()
        mask_rewards[action["idx"]] = reward
        
        # params matrix
        # mask_params = online_action["params"][random.choices(range(0, batch_size), k=mask_rewards.shape[0])]
        # mask_params[action["idx"]] = action["params"]
        # mask_params = mask_params.transpose([1, 0])
        
        # # orientations_map_random = np.dot(
        # #     R.from_euler("ZYX", rot_random).as_matrix(), R.from_euler("xyz", [-np.pi, 0, 0]).as_matrix()
        # # )
        # # orientations_map_random = orientations_map_random.reshape(orientations_map.shape)
        # # normals_random = -orientations_map_random[:, :, :, 2]

        # action["mask_idxs"] = mask_idxs
        # action["mask_rewards"] = mask_rewards
        # action["mask_params"] = mask_params

        # new_replay_memory.add(
        #     obs0=obs, action=action, reward=reward, obs1=obs_next, terminal1=done
        # )
        # new_replay_memory.save(new_replay_memory_path)
            

