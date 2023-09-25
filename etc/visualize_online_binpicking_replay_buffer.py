import argparse
import functools
import itertools
import os
import random
from copy import deepcopy

import time
import cv2
import gym
import numpy as np
import rospy
from gym.envs.registration import register
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation as R
from tami_gym_ros.task_envs.bin_picking_sim_task import BinPickingEnv
from tami_gym_ros.utils.utils import load_bin
from tami_rl_agent.algorithms.consac import ConSAC
from tami_rl_agent.utils.geometry import angle_between_vectors, location_pixel, point_in_hull, sample_hull
from tami_rl_agent.utils.object_detection import ChangeDetector
from tami_rl_agent.utils.visualization import (
    generate_grasp_on_height_map_image,
    rotate_image,
    rotate_pixel,
    visualize_object_detection,
)

from tami_trainer.simple_trainer import SimpleTrainer
from tami_rl_agent.utils.replay_memory import ReplayMemory

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

def display_imgs(input_image, heatmap, title="image"):
    rgb_title = f"{title}_rgb"
    depth_title = f"{title}_depth"
    normalized_title = f"{title}_normalized"
    heatmap_title = f"{title}_heatmap"

    rgb = input_image[:, :, 1:4]
    rgb = rgb.astype(np.uint8)
    depth = input_image[:, :, :1]
    depth = (255 * (depth - depth.min()) / (depth.max() - depth.min())).astype(np.uint8)

    normalized = input_image[:, :, 4:]
    normalized = (255 * (normalized - normalized.min()) / (normalized.max() - normalized.min())).astype(np.uint8)

    cv2.namedWindow(rgb_title, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_title, cv2.WINDOW_NORMAL)
    cv2.namedWindow(normalized_title, cv2.WINDOW_NORMAL)
    cv2.namedWindow(heatmap_title, cv2.WINDOW_NORMAL)
    cv2.imshow(rgb_title, rgb)
    cv2.imshow(depth_title, depth)
    cv2.imshow(normalized_title, normalized)
    cv2.imshow(heatmap_title, heatmap)

    cv2.resizeWindow(rgb_title, 800, 800)
    cv2.resizeWindow(depth_title, 800, 800)
    cv2.resizeWindow(normalized_title, 800, 800)
    cv2.resizeWindow(heatmap_title, 800, 800)
    # here it should be the pause
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()

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
        "-bc",
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
    args = parser.parse_args()

    # memory_path_p = os.path.expanduser(
    #     f"~/amira_experiments/memories/{args.env_id}/{args.robot}/{args.config_file}/{args.memory_path}"
    # )
    
    replay_memory = ReplayMemory(capacity=int(1e6), default_sampler="random")

    replay_memory.load(os.path.expanduser(f"~/amira_experiments/memories/{args.memory_path}"))
    replay_memory.mark_as_synced()

    label = 'heatmap'
    
    # batch = 
    for idx in range(len(replay_memory)):
        print (f"idx: {idx}, ")
        
        
        time.sleep(0.1)
        obs, action, reward, obs_next, done, _ = replay_memory.memory[idx]
        flatten_obs = obs.reshape(7, -1)
        idx_obs = flatten_obs[:, action['idx']]
        
        print (f"idx_obs: {idx_obs.mean()}")
    
        # display_imgs(np.transpose(obs[:, :, ...], (1, 2, 0)), np.transpose(action['heatmaps'][0], (0, 1, 2)), title="obs")
        
        # obs_smooth = obs.copy()
        # _, height, width = obs_smooth.shape
        
        # # preprocess data
        # # Find the zero pixels
        # obs_smooth_temp = obs_smooth[1:4, ].sum(0)
        # obs_smooth_zero_pixels = np.where(obs_smooth_temp == 0.)
        # height_idxes, width_idxes = obs_smooth_zero_pixels
        # range_pixels = 3
        
        # # Iterate over the zero pixels
        # for i, j in zip(height_idxes, width_idxes):
        #     # Find the non-zero pixels within the 4x4 window
        #     i_start = i-range_pixels if i-range_pixels > 0 else i
        #     i_end = i+range_pixels if i+range_pixels < height else height
            
        #     j_start = j-range_pixels if j-range_pixels > 0 else j
        #     j_end = j+range_pixels if j+range_pixels < width else width
            
        #     obs_smooth_patch = obs_smooth[:, i_start:i_end, j_start:j_end]
        #     non_zero_obs_smooth_patch = obs_smooth_patch[1:4].sum(0).nonzero()

        #     # Compute the mean of the non-zero pixels
        #     # if len(non_zero_obs_smooth_patch[0]) > 0:
        #         # Replace the zero pixel with the mean
        #         # obs_smooth[1:4, i, j] = obs_smooth_patch[:, non_zero_obs_smooth_patch[0], non_zero_obs_smooth_patch[1]].min(-1)
        #         # obs_smooth[1:4, i, j] = obs_smooth_patch[:, non_zero_obs_smooth_patch[0], non_zero_obs_smooth_patch[1]].max(-1)
            
        #     obs_smooth[1:4, i, j] = obs_smooth_patch[1:4].mean(-1).mean(-1)
        #     # obs_smooth[1:, i, j] = obs_smooth_patch[:1].mean(-1).mean(-1)
        #     # obs_smooth[4:, i, j] = obs_smooth_patch[4:].mean(-1).mean(-1)
        #     # obs_smooth[1:4, i, j] = obs_smooth_patch.min(-1).min(-1)
                    
        # display_imgs(np.transpose(obs_smooth[:, :, ...], (1, 2, 0)), np.transpose(action['heatmaps'][0], (0, 1, 2)), title="obs_smooth")
    

    # if args.add_offline_data:
    #     bin_path = f"{args.config_model_path}/config/{args.bin_config_file}.yaml"
    #     bin_hull, bbox_center, bbox_hsize = load_bin(os.path.expanduser(bin_path))
    #     hm_size = 2 * bbox_hsize
    #     hm_resolution = 0.0035
    #     data_path = os.path.expanduser(f"~/amira_experiments/datasets/{args.bin_config_file}/{args.data_path}")
    #     data_paths = [
    #         os.path.join(data_path, o) for o in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, o))
    #     ]
    #     agent.load_offline_memory(data_paths, bbox_center, hm_size, hm_resolution, mark_synced=True)

    # This is needed if using python3.9 and OpenCV 4.5.5
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    # run_trainer(
    #     env_picking=env,
    #     agent_consac=agent,
    #     steps=args.total_steps,
    #     model_path=model_path_p,
    #     memory_path=memory_path_p,
    #     debug_output_path=args.debug_output_path,
    #     without_online_sample=args.without_online_sample,
    # )
    # env.close()
