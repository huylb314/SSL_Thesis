import argparse
import multiprocessing
import os
from functools import partial
import logging
from tqdm import tqdm

import cv2
import torch
import random
import json
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion, convolve
from scipy.spatial.transform import Rotation as R
from tami_gym_ros.utils.utils import load_bin, load_meshes, transform_gripper_meshes
from tami_rl_agent.utils.ssl_utils import create_eval_losses
from tami_rl_agent.algorithms.consac import ConSAC
from tami_rl_agent.ssl_models.fixmatch.consac_fixmatch import ConSACFixMatch
from tami_rl_agent.ssl_models.online.consac_online import ConSACOnline
from tami_rl_agent.ssl_models.offline.consac_offline import ConSACOffline
from tami_rl_agent.ssl_models.flexmatch.consac_flexmatch import ConSACFlexMatch
from tami_rl_agent.ssl_models.flexmatch.consac_contexture_flexmatch import ConSACContextureFlexMatch
from tami_rl_agent.ssl_models.freematch.consac_freematch import ConSACFreeMatch
from tami_rl_agent.ssl_models.freematch.consac_contexture_freematch import ConSACContextureFreeMatch
from tami_rl_agent.ssl_models.softmatch.consac_softmatch import ConSACSoftMatch
from tami_rl_agent.ssl_models.softmatch.consac_contexture_softmatch import ConSACContextureSoftMatch
from tami_rl_agent.ssl_models.softmatch.consac_softmatch_new import ConSACSoftMatchNew
from tami_rl_agent.ssl_models.softmatch.consac_contexture_softmatch_new import ConSACContextureSoftMatchNew
from tami_rl_agent.utils.geometry import array_pcl_to_pcl, transform_xyz_quat_to_matrix
from tami_rl_agent.utils.preprocess_utils import DilateDepthHoles, DepthSmoothing, DepthToPoints, AverageNormals, NormalsStd, NormalsDiff
import math

# # Panda4 Zivid camera config
# extrinsic = [0.50263718, -0.25443212, 0.79777664, 0.70649618, -0.69977101, -0.08671922, -0.06052648]
# intrinsic = [1280, 720, 917.347412109375, 917.256591796875, 643.0258178710938, 378.55291748046875]
# extrinsic_matrix = transform_xyz_quat_to_matrix(extrinsic)
# intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(*intrinsic).intrinsic_matrix

# Tuebingen Realsense setup
extrinsic = [0.3947, 0.0191, 0.8936, -0.6964, 0.7174, 0.0201, 0.0032]
intrinsic = [1280, 720, 922.8173828125, 922.1414794921875, 615.5975341796875, 354.6059875488281]
# x,y,z,qx,qy,qy,qw
extrinsic_matrix = transform_xyz_quat_to_matrix(extrinsic)
intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(*intrinsic).intrinsic_matrix

# normalize
MAX_REWARD = 0.6132314205169678
MIN_REWARD = 0.0

MEDIAN_SIZE = 2
CLOSE_SIZE = 3
BLUR_BOX_SIZE = 3
BLUR_GAUSSIAN_SIZE = 3
SUCTION_SIZE = 6
FILTER_SIZES = [2, 4]
Q_THRESHOLD = 0.2

def orthographic_projection(pcl, orig, size, resolution, flipud=False):
    """
    Project a point cloud onto the XY-hyperplane clipped to a region of interest bounding box
    Args:
        flipud: flip the output so that positive y direction would go upwards
        pcl: (N, 6) ndarray
        pixel_density: the number of pixels per a unit of measurement (int)
        bbox_center: the center of region
        bbox_half_sizes: 3d array representing the distance from center of the region in each dimension
        take_max_z: bool which specified whether to take the max z value of all points that fall in a pixel or take one
        of the z values arbitrarily (slightly faster)

    Returns:
        an image with 6 channels (x, y, z, r, g, b)
    """
    n_voxels = size / resolution
    n_voxels = (np.ceil(n_voxels / 8) * 8).astype(np.uint16)

    lb = orig
    ub = orig + size
    reduced_pcl = pcl
    reduced_pcl = pcl[np.all(pcl[:, :3] >= lb, axis=1) & np.all(pcl[:, :3] <= ub, axis=1), :]
    xy_image = np.zeros((n_voxels[1], n_voxels[0], pcl.shape[1]))
    if pcl.shape[1] > 8:
        xy_image[:, :, 8] = 1
    xy_image[:, :, 2] = lb[2]
    z_image = lb[2] * np.ones((n_voxels[1], n_voxels[0], n_voxels[2]))
    xx, yy = np.meshgrid(range(n_voxels[0]), range(n_voxels[1]), indexing="xy")
    xy_image[:, :, 0] = lb[0] + np.array(0.5 + xx, dtype=np.float32) * resolution
    xy_image[:, :, 1] = lb[1] + np.array(0.5 + yy, dtype=np.float32) * resolution
    pixels = np.array(((reduced_pcl[:, :3] - lb[:3]) // resolution), dtype=np.int32)
    xy_image[pixels[:, 1], pixels[:, 0], 2:] = reduced_pcl[:, 2:]
    z_image[pixels[:, 1], pixels[:, 0], pixels[:, 2]] = reduced_pcl[:, 2]
    xy_image[:, :, 2] = np.max(z_image, axis=2)
    if flipud:
        xy_image = np.flipud(xy_image)
    return xy_image


def visualize_grasp_affordance(pcl: np.ndarray, reward: np.ndarray) -> None:
    """Visualize the pcl with the good grasp points masked in red color

    Args:
        pcl (np.ndarray): [description]
        reward (np.ndarray): [description]
    """
    r = np.expand_dims(reward.flatten(), 1)
    # pos_idxs = np.where(r > 0)[0]
    pcl_mask = pcl.copy()
    # pcl_mask[list(pos_idxs), 3:] = np.array([255, 0, 0]).astype(np.uint8)
    pcl_mask[:, 3:6] = (1 - r) * pcl_mask[:, 3:6] + r * np.array([255, 0, 0])
    pcd_mask = array_pcl_to_pcl(pcl_mask)
    o3d.visualization.draw_geometries([pcd_mask], window_name="Grasp Affordance")


def display_imgs(rgb, depth, title=""):
    cv2.namedWindow(f"{title}_Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow(f"{title}_Rgb", cv2.WINDOW_NORMAL)
    cv2.imshow(f"{title}_Depth", (255 * (depth - depth.min()) / (depth.max() - depth.min())).astype(np.uint8))
    cv2.imshow(f"{title}_Rgb", rgb.astype(np.uint8))
    cv2.resizeWindow(f"{title}_Depth", 800, 800)
    cv2.resizeWindow(f"{title}_Rgb", 800, 800)
    cv2.waitKey(10)


def display_img(img, title=""):
    cv2.namedWindow(f"{title}_img", cv2.WINDOW_NORMAL)
    cv2.imshow(f"{title}_img", (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8))
    cv2.resizeWindow(f"{title}_img", 800, 800)
    cv2.waitKey(10)

def display_heightmap(hm: np.ndarray, title="") -> None:
    """Disaplay images for the rgb and z maps of a hei-ghtmap

    Args:
        hm (np.ndarray): [description]
    """
    cv2.namedWindow(f"{title}_z", cv2.WINDOW_NORMAL)
    cv2.namedWindow(f"{title}_rgb", cv2.WINDOW_NORMAL)
    cv2.imshow(
        f"{title}_z", (255 * (hm[:, :, 2] - hm[:, :, 2].min()) / (hm[:, :, 2].max() - hm[:, :, 2].min())).astype(np.uint8)
    )
    cv2.imshow(f"{title}_rgb", hm[:, :, 3:6].astype(np.uint8))
    cv2.resizeWindow(f"{title}_z", 800, 800)
    cv2.resizeWindow(f"{title}_rgb", 800, 800)
    cv2.waitKey(10)


def display_random_grasp(
    hm_featured: np.ndarray, orientations_map: np.ndarray, q_map: np.ndarray, pcl: np.ndarray, gripper_meshes: dict
) -> None:
    """visualizes grasp in pcl for a ramdomly chosen pixel of the heightmap

    Args:
        hm_featured (np.ndarray): [description]
        orientations_map (np.ndarray): [description]
        q_map (np.ndarray): [description]
        pcl (np.ndarray): [description]
        gripper_meshes (dict): [description]
    """
    grasp_idx = np.random.choice(list(range(len(q_map.flatten()))), p=(q_map.flatten() / q_map.sum()))
    grasp_idx = np.argmax(q_map.flatten())
    grasp_pixel = np.unravel_index(grasp_idx, hm_featured.shape[:2])
    translation = hm_featured[grasp_pixel][:3]
    orientation = orientations_map[grasp_pixel]
    T_r = np.eye(4)
    T_r[:3, :3] = orientation
    T_r[:3, 3] = translation
    meshes = []
    for mesh_ in gripper_meshes.values():
        meshes.append(o3d.geometry.TriangleMesh(mesh_).transform(T_r))
    # pcl_ = pcl[np.where(np.linalg.norm(pcl[:, :3] - translation, axis=1) < 0.15)]
    o3d.visualization.draw_geometries([array_pcl_to_pcl(pcl)] + meshes, window_name="Random (Good) Grasp")


def calculate_scene_com(pcl: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """Calculate the center of mass of the scene based on the annotations locations

    Args:
        pcl (np.ndarray): [description]
        reward (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    r = reward.flatten()
    pcl_masked = pcl[np.where(r > 0)[0], :3]
    return np.mean(pcl_masked, axis=0)


def calculate_grasp_orientation_map_from_normals(normals: np.ndarray) -> np.ndarray:
    """Calculate orientation matrix for each pixel of the normal map

    Args:
        normals (np.ndarray): The normals map

    Returns:
        np.ndarray: orientation map
    """
    G_z = -normals
    G_z /= np.linalg.norm(G_z, axis=2)[:, :, np.newaxis].repeat(3, axis=2)
    G_x = np.stack([np.ones(normals.shape[:2]), np.zeros(normals.shape[:2]), -G_z[:, :, 0] / G_z[:, :, 2]]).transpose(
        1, 2, 0
    )
    G_x /= np.linalg.norm(G_x, axis=2)[:, :, np.newaxis].repeat(3, axis=2)
    G_y = np.cross(G_z, G_x)
    o_matrix = np.stack((G_x, G_y, G_z), axis=2).transpose(0, 1, 3, 2)
    # The code below is probably redundant
    # for i in range(O.shape[0]):
    #     for j in range(O.shape[1]):
    #         if np.linalg.det(O[i, j]) < 0:
    #             O[i, j, :3, 1] *= -1
    return o_matrix


def calc_pixel_wise_normals(depth_pts: np.ndarray):
    depth_pts = np.transpose(depth_pts, axes=[2, 0, 1])
    normals_unscaled = np.cross(np.gradient(depth_pts, axis=-1), np.gradient(depth_pts, axis=-2), axisa=0, axisb=0)
    mult = [np.multiply(normals_unscaled[:, :, d], normals_unscaled[:, :, d]) for d in range(normals_unscaled.shape[2])]
    norm = np.sqrt(np.sum(mult, axis=0))
    with np.errstate(invalid="ignore"):  # warning for missing depth pixels (zero-by-zero divide), expected and ok
        return normals_unscaled / np.dstack((norm, norm, norm))


def calc_planar_reward(pcl, normals, obj_mask, suction_size=0.03):
    reward = np.zeros_like(obj_mask)
    for i in np.where(obj_mask)[0]:
        normal = normals[i]
        pnt = pcl[i, :3]
        pcl_translated = pcl[:, :3] - pnt
        N_z = normal
        N_x = np.array([1, 0, -N_z[0] / N_z[1]])
        N_x /= np.linalg.norm(N_x)
        N_y = np.cross(N_z, N_x)
        rot = np.stack((N_x, N_y, N_z)).T
        pcl_transformed = np.dot(np.linalg.inv(rot), pcl_translated.T).T
        pcl_suction = pcl_transformed[np.where(np.linalg.norm(pcl_transformed[:, :2], axis=1) < suction_size / 2.0)]
        flatness_score = (pcl_suction[:, 2] ** 2).mean()
        print(f"Flatness score:{flatness_score}")
        reward[i] = 1 - flatness_score
    reward /= reward.max()
    return reward


def calc_angles_between_normals(normals, pcl, mask, cup_size):
    points = pcl[np.where(mask)[0], :3]
    normals = normals[np.where(mask)[0], :]
    points_min = points.min(axis=0)
    points_idxs = ((points - points_min) / 0.001).astype(np.uint32)
    idx_voxels = -np.ones((1000, 1000, 1000))
    idx_voxels[points_idxs[:, 0], points_idxs[:, 1], points_idxs[:, 2]] = range(len(points))
    reward = np.zeros_like(mask, dtype=np.float32)
    for i, point in enumerate(points):
        idxs = ((point - points_min) / 0.001).astype(np.int32)
        lb = np.maximum(idxs[:2] - 20, 0)
        ub = np.minimum(idxs[:2] + 20, idx_voxels.shape[0] - 1)
        prox_set = np.unique(idx_voxels[lb[0] : ub[0], lb[1] : ub[1], :])
        prox_set = np.delete(prox_set, np.where(prox_set < 0)).astype(np.uint32)
        dists = np.linalg.norm(points[prox_set] - point, axis=1)
        prox_set = prox_set[np.where(dists < cup_size / 2)]
        prox_normals = normals[prox_set, :]
        normals_mult = np.dot(prox_normals, prox_normals.T)
        angles = np.arccos(np.clip(normals_mult, -1, 1))
        avg_angle = angles.mean()
        reward[i] = 1 - avg_angle / np.pi
        print(f"reward[{i}/{len(points)}]={reward[i]}")
    return reward


def calc_normals_based_reward(hm, cup_size, hm_resolution):
    reward = np.zeros(hm.shape[:2])
    for h in range(hm.shape[0]):
        for w in range(hm.shape[1]):
            n_pixels = int(cup_size / (2 * hm_resolution) + 0.5)
            points = hm[
                max(0, h - n_pixels) : min(hm.shape[0], h + n_pixels),
                max(0, w - n_pixels) : min(hm.shape[1], w + n_pixels),
                :3,
            ]
            points = points.reshape(-1, 3)
            dists = np.linalg.norm(points - hm[h, w, :3], axis=1)
            normals = hm[
                max(0, h - n_pixels) : min(hm.shape[0], h + n_pixels),
                max(0, w - n_pixels) : min(hm.shape[1], w + n_pixels),
                6:9,
            ]
            normals = normals.reshape(-1, 3)
            idxs = dists < (cup_size / 2.0)
            normals = normals[idxs]
            normals_mult = np.dot(normals, normals.T)
            angles = np.arccos(np.clip(normals_mult, -1, 1))
            avg_angle = angles.mean()
            reward[h, w] = 1 - (avg_angle / np.pi) ** (0.25)
    return reward


def average_normals(normals, suction_size):
    avg_normals = (
        convolve(normals, np.ones((suction_size, suction_size, 1)), mode="constant", cval=0.0) / suction_size**2
    )
    return avg_normals


def mask_pcl_from_other_masked_pcl(pcl, pcl_masked, mask):
    obj_mask = np.zeros(len(pcl))
    obj_pcl = pcl_masked[np.where(mask)[0], :]
    pcd = array_pcl_to_pcl(pcl)
    obj_pcd = array_pcl_to_pcl(obj_pcl)
    dists = pcd.compute_point_cloud_distance(obj_pcd)
    print(dists)
    obj_mask[np.where(np.asarray(dists) < 0.005)] = 1
    print(f"Found {obj_mask.sum()} object points!!!")
    return obj_mask


def compute_obs_action_r(
    path: str,
    bbox_center: np.ndarray,
    hm_size: np.ndarray,
    hm_resolution: float,
    recalc_reward=False,
    debug=False
):
    rgb = cv2.imread(os.path.join(path, "rgb.png"))
    depth = cv2.imread(os.path.join(path, "depth.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    normals = cv2.imread(os.path.join(path, "normals.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    obj_mask = cv2.imread(os.path.join(path, "obj_mask.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    pcl = cv2.imread(os.path.join(path, "pcl.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    pcl_extended = np.hstack((pcl, normals, obj_mask.astype(np.float32)))

    # Create heightmap

    # Don't center
    hm_origin = bbox_center.copy() - hm_size / 2
    # but add small random shift to the center: Dont use stored planar_reward if ENABLE this

    hm_featured = orthographic_projection(pcl_extended, hm_origin, hm_size, hm_resolution, flipud=True)
    hm_featured[:, :, -1] = binary_erosion(hm_featured[:, :, -1], structure=np.ones((3, 3)), iterations=2)
    planar_reward = None
    if not recalc_reward and os.path.exists(os.path.join(path, "q_map.exr")):
        planar_reward = cv2.imread(os.path.join(path, "q_map.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if planar_reward.shape != hm_featured.shape[:2]:
            planar_reward = None  # discard stored q_map if its shape does not fit new setting
    if planar_reward is None:
        planar_reward = calc_normals_based_reward(hm_featured, 0.04, hm_resolution)
        cv2.imwrite(os.path.join(path, "q_map.exr"), planar_reward.astype(np.float32))

    # DEBUG IMAGES
    if debug:
        display_heightmap(hm_featured, "original")
        display_imgs(rgb, depth, "original")
        
    # visualize and preprocessing
    # Fill missing depth information to avoid incorrect data and smoothing artifacts
    hm_featured[:, :, 2] = DilateDepthHoles(hm_featured[:, :, 2], filter_sizes=FILTER_SIZES, adaptive_scaling=False)
    
    # display_heightmap(hm_featured, "processed")
    
    # Improve quality of depth image for further processing
    hm_featured[:, :, 2] = DepthSmoothing(
        depth=hm_featured[:, :, 2], 
        median_size=MEDIAN_SIZE,
        close_size=CLOSE_SIZE,
        blur_box_size=BLUR_BOX_SIZE,
        blur_gaussian_size=BLUR_GAUSSIAN_SIZE,
        adaptive_scaling=False,
    )
    # Filter the depth image based on background and range
    
    # Calculate standard deviation of normal vectors to find edges
    normal_averaged, normal_squared = AverageNormals(hm_featured[:, :, 6:9], suction_size=SUCTION_SIZE, do_squared=True)
    std_normal = NormalsStd(normal_averaged, normal_squared)
    diff_normal = NormalsDiff(hm_featured[:, :, 6:9], normal_averaged)
    hm_featured[:, :, 6:9] = normal_averaged

    # DEBUG MASK
    if debug:
        visualize_grasp_affordance(
            hm_featured[:, :, :6].reshape(-1, 6), planar_reward.flatten() * hm_featured[:, :, -1].flatten()
        )
    # DEBUG HEIGHTMAP
    if debug:
        pcd = array_pcl_to_pcl(hm_featured[:, :, :9].reshape(-1, 9))
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        display_heightmap(hm_featured)

    # Create obs
    obs = hm_featured[:, :, 2:9].transpose(2, 0, 1).copy()
    # Create action params (gripper rotation)
    normals_map = hm_featured[:, :, 6:9]
    orientations_map = calculate_grasp_orientation_map_from_normals(normals_map)
    # Calculate rotation from base pose to desirable grasp pose
    base_gripper_pose = R.from_euler("xyz", [-np.pi, 0, 0]).as_matrix()
    rot = np.dot(orientations_map, np.linalg.inv(base_gripper_pose))
    rot = rot.reshape(np.prod(rot.shape[:2]), 3, 3)
    rot = np.flip(R.from_matrix(rot).as_euler("ZYX"), axis=1)
    # Create affordance supervision
    q_map = hm_featured[:, :, -1] * planar_reward
    # display_img(hm_featured[:, :, -1] * planar_reward, "planar")
    # q_map = hm_featured[:, :, -1] * (1. - std_normal)
    # display_img(hm_featured[:, :, -1] * (1. - std_normal), "std")
    # display_img(((hm_featured[:, :, -1] * (1. - std_normal)) > 0.5).astype(np.float32), "0.5std")
    # display_img(hm_featured[:, :, -1] * (1. - diff_normal), "diff")
    
    # q_feas = agent.get_feasibility_map(obs, rot.T.copy()).reshape(q_map.shape[:2])
    # DEBUG GRASP
    if debug:
        display_random_grasp(hm_featured, orientations_map, q_map, pcl, gripper_meshes)

    # Generate random orientations
    rot_random = np.random.uniform(-0.785, 0.785, size=rot.shape)
    # q_feas_random = agent.get_feasibility_map(obs, rot_random.T.copy()).reshape(q_map.shape[:2])
    orientations_map_random = np.dot(
        R.from_euler("ZYX", rot_random).as_matrix(), R.from_euler("xyz", [-np.pi, 0, 0]).as_matrix()
    )
    orientations_map_random = orientations_map_random.reshape(orientations_map.shape)
    normals_random = -orientations_map_random[:, :, :, 2]

    # Calculate reward based on the angle between normal and grasp pose (small is good)
    angles_grasp_normals = np.arccos(np.clip(np.sum(normals_random * hm_featured[:, :, 6:9], axis=2), -1.0, 1.0))
    q_map_random = q_map * np.exp(-2 * angles_grasp_normals**2).astype(np.float32)

    # DEBUG GRASP RANDOM
    if debug:
        display_random_grasp(hm_featured, orientations_map_random, q_map_random, pcl, gripper_meshes)

    # Normalize actions
    rot = agent.normalize_action(rot)
    rot_random = agent.normalize_action(rot_random)
    
    # Binarize q_map
    q_map = (q_map > Q_THRESHOLD).astype(float) * 1.0
    q_map_random = (q_map_random > Q_THRESHOLD).astype(float) * 1.0

    return obs.copy(), rot.copy().T, rot_random.copy().T, q_map.copy(), q_map_random.copy()


def train_model_offline(
    agent: ConSAC,
    train_epochs: int,
    model_path: str,
    data_paths: list,
    bbox_center: np.ndarray,
    hm_size: np.ndarray,
    hm_resolution: float,
    gripper_meshes: dict,
    batch_size: int,
    recalc_reward=False,
    debug: bool = False,
    normalize_rewards: bool=False,
    freq_epoch: int=100,
    w_offline: float=0,
) -> None:
    """Train grasping model

    Args:
        agent (ConSAC): The grasp agent
        train_epochs (int): the number of training epochs to perform
        model_path (str): the path where the grasp model state is saved
        data_path (str): The path to the Acronym data containing the scenes
        bbox_center (np.ndarray): the center of the scene bounding box
        hm_size (np.ndarray): the 3D size of the heightmap generated from the orthographic projection
        hm_resolution (float): The pixel size of the heightmap generated from the orthographic projection
        gripper_meshes (dict): map gripper mesh name to o3d.Trimesh transformed to TCP frame
        debug (bool, optional): Enable visualization to validate corretness. Defaults to False.
    """
    pool = multiprocessing.Pool()
    agent_epochs = agent.epochs
    for epoch in range(agent_epochs, train_epochs + 1):
        logging.info(f"=================== EPOCH {epoch} =====================")
        np.random.shuffle(data_paths)
        if len(data_paths) % batch_size != 0:
            full_epoch = batch_size * (len(data_paths) // batch_size)
        else:
            full_epoch = len(data_paths)

        batch_paths_list = np.array_split(data_paths[:full_epoch], len(data_paths) // batch_size)
        if len(data_paths) % batch_size != 0:
            batch_paths_list.append(data_paths[full_epoch:])
        my_func = partial(
            compute_obs_action_r,
            bbox_center=bbox_center,
            hm_size=hm_size,
            hm_resolution=hm_resolution,
            recalc_reward=(epoch == 0 and recalc_reward),
            debug=debug
        )
        for id, batch_paths in enumerate(tqdm(batch_paths_list)):
            obs_batch = list()
            reward_batch = list()
            action_batch = list()
            action_rand_batch = list()
            reward_rand_batch = list()
            if w_offline > 0:
                obs_a_r = pool.map(my_func, batch_paths)
                for b in obs_a_r:
                    obs_batch.append(b[0])
                    action_batch.append(b[1])
                    action_rand_batch.append(b[2])
                    reward_batch.append(b[3])
                    reward_rand_batch.append(b[4])
                obs_batch = np.stack(obs_batch, axis=0)
                reward_batch = np.stack(reward_batch, axis=0)
                action_batch = np.stack(action_batch, axis=0)
                reward_rand_batch = np.stack(reward_rand_batch, axis=0)
                action_rand_batch = np.stack(action_rand_batch, axis=0)

            agent.update_ssl_critic_full_img(obs_batch, action_batch, reward_batch, batch_id=id, epoch=epoch)
            agent.update_ssl_actor_full_img(obs_batch, action_batch, reward_batch)
            agent.update_ssl_critic_full_img(obs_batch, action_rand_batch, reward_rand_batch, batch_id=None, epoch=epoch)
            agent.update_ssl_actor_full_img(obs_batch)
            
            if ((epoch % freq_epoch == 0 and epoch > 0) or (epoch == train_epochs)) and id == (len(batch_paths_list) - 1):
                save_path = os.path.join(model_path, str(epoch))
                agent.save_state(save_path)


def train_model(
    agent: ConSAC,
    train_epochs: int,
    model_path: str,
    data_paths: list,
    bbox_center: np.ndarray,
    hm_size: np.ndarray,
    hm_resolution: float,
    gripper_meshes: dict,
    batch_size: int,
    recalc_reward=False,
    debug: bool = False,
    normalize_rewards: bool=False,
    freq_epoch: int=100,
    w_offline: float=0,
) -> None:
    """Train grasping model

    Args:
        agent (ConSAC): The grasp agent
        train_epochs (int): the number of training epochs to perform
        model_path (str): the path where the grasp model state is saved
        data_path (str): The path to the Acronym data containing the scenes
        bbox_center (np.ndarray): the center of the scene bounding box
        hm_size (np.ndarray): the 3D size of the heightmap generated from the orthographic projection
        hm_resolution (float): The pixel size of the heightmap generated from the orthographic projection
        gripper_meshes (dict): map gripper mesh name to o3d.Trimesh transformed to TCP frame
        debug (bool, optional): Enable visualization to validate corretness. Defaults to False.
    """
    agent_epochs = agent.epochs
    num_batch = len(agent.memory_dl)
    num_eval_batch = len(agent.eval_memory_dl)
    for epoch in range(agent_epochs, train_epochs + 1):
        logging.info(f"=================== EPOCH {epoch} =====================")
        # tensorboard: losses and logs
        losses, logs = { 
            "critic/critic_loss": 0.,
            "critic/online_critic_loss": 0.,
            "critic/ssl_loss": 0.,
            "critic/pessimistics_q": 0.,
            "actor/online_actor_q": 0.,
        }, {
            "ssl/q1_ssl_confidence_neg": 0.,
            "ssl/q1_ssl_confidence_pos": 0.,
            "ssl/negative": 0.,
            "ssl/positive": 0.,
            "ssl/negative_num": 0.,
            "ssl/positive_num": 0.,
        }
        for idx, sampled_batch in tqdm(enumerate(agent.memory_dl), total=num_batch):
            agent.update_ssl_critic_full_img(losses, logs, sampled_batch, batch_id=idx, epoch=epoch, num_batch=num_batch)
            agent.update_ssl_actor_full_img(losses, logs, sampled_batch, num_batch=num_batch)
        
        # Evaluation Step
        if agent.eval_memory_size > 0 and epoch % agent.evaluation_freq == 0:
            losses = create_eval_losses(losses, tag="critic/", eval_points=["1", "10", "100", "full"])
            losses = create_eval_losses(losses, tag="critic/EMA_", eval_points=["1", "10", "100", "full"])

            for eval_idx, sampled_batch_eval in tqdm(enumerate(agent.eval_memory_dl), total=num_eval_batch):
                agent.evaluation_critic(losses, logs, sampled_batch_eval, visualize=True, tag="critic/", batch_id=eval_idx, epoch=epoch, num_eval_batch=num_eval_batch)
                agent.ema.apply_shadow()
                agent.evaluation_critic(losses, logs, sampled_batch_eval, visualize=False, tag="critic/EMA_", batch_id=None, epoch=epoch, num_eval_batch=num_eval_batch)
                agent.ema.restore()

        # logging
        for name, value in losses.items():
            agent._summary_writer.add_scalar(f"{name}", value if isinstance(value, float) else value.cpu().detach().numpy(), epoch)
        # debug: logging
        for name, value in logs.items():
            agent._summary_writer.add_scalar(f"{name}", value if isinstance(value, float) else value.cpu().detach().numpy(), epoch)
        
        # if ((epoch % freq_epoch == 0 and epoch > 0) or (epoch == train_epochs)):
        #     save_path = model_path
        #     agent.save_state(save_path)


def evaluate_model(
    agent: ConSAC,
    data_paths: list,
    bbox_center: np.ndarray,
    hm_size: np.ndarray,
    hm_resolution: float,
    gripper_meshes: dict,
    visualize: bool,
) -> None:
    """Evaluate the trained grasping model

    Args:
        agent (ConSACSSL): The grasp agent
        data_paths (list): the path of the data directories
        bbox_center (np.ndarray): the center of the scene bounding box
        hm_size (np.ndarray): the 3D size of the heightmap generated from the orthographic projection
        hm_resolution (float): The pixel size of the heightmap generated from the orthographic projection
        gripper_meshes (dict): map gripper mesh name to o3d.Trimesh transformed to TCP frame
    """
    for path in data_paths:
        print(path)
        pcl = cv2.imread(os.path.join(path, "pcl.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        normals = cv2.imread(os.path.join(path, "normals.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        pcl_extended = np.hstack((pcl, normals))

        # scene_poc = calculate_scene_com(pcl, obj_mask)
        # hm_origin = scene_poc[:3] - hm_size / 2
        hm_origin = bbox_center.copy() - hm_size / 2
        hm = orthographic_projection(pcl_extended, hm_origin, hm_size, hm_resolution, flipud=True)
        obs = hm[:, :, 2:9].transpose(2, 0, 1)
        action = agent.select_action(
            dict(visual=dict(image=obs.copy(), loc_map=hm[:, :, :3].transpose(2, 0, 1))), train=False
        )
        T_r = np.eye(4)
        euler = action["params_scaled"]
        orientation = np.dot(
            R.from_euler("ZYX", np.flip(euler)).as_matrix(), R.from_euler("xyz", [-np.pi, 0, 0]).as_matrix()
        )
        T_r[:3, :3] = orientation
        T_r[:3, 3] = action["location"]
        meshes = transform_gripper_meshes(gripper_meshes, gripper_transform=T_r, trans_quat=False)
        if visualize:
            o3d.visualization.draw_geometries([array_pcl_to_pcl(pcl)] + meshes, window_name="Best Grasp")


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline SSL (FixMatch) Bin Picking Trainer")
    parser.add_argument("-e", "--env_id", dest="env_id", type=str, default="BinPicking-v1", help="The task env")
    parser.add_argument(
        "-sparse",
        "--sparse_reward",
        dest="sparse_reward",
        type=bool,
        default=True,
        help="Whether combine offline data with sparse reward feedback data"
    )
    parser.add_argument(
        "-r", "--robot", dest="robot", type=str, default="panda_vacuum", help="The robot used for bin picking"
    )

    parser.add_argument(
        "-gc",
        "--gripper_conf",
        dest="gripper_config_file",
        type=str,
        default="panda_hfa_suction_gripper",
        help="The name of the env config file",
    )

    parser.add_argument(
        "-bc",
        "--bin_conf",
        dest="bin_config_file",
        type=str,
        default="2bins_real_tue",
        help="The name of the env config file",
    )

    parser.add_argument("--log_dir", dest="log_dir", type=str, default="real", help="The path to save the logs")
    parser.add_argument("--vis_path", dest="visualization_path", type=str, default="vis", help="The path to save visualization of each evaluation step")

    parser.add_argument("--debug", action="store_true", dest="debug")

    parser.add_argument(
        "--data_path", dest="data_path", type=str, default="near_wall_2", help="The path to save the img data"
    )
    parser.add_argument("--eval_path",
                        dest="eval_path",
                        type=str,
                        default="",
                        help="The path to save the memory")
    
    parser.add_argument(
        "--mem_path",
        dest="memory_path",
        type=str,
        default="tue_suction",
        help="The path to save the memory",
    )
    parser.add_argument(
        "--eval_mem_path",
        dest="eval_memory_path",
        type=str,
        default="",
        help="The path to load the evaluation memory",
    )
    parser.add_argument(
        "--full_data_path", dest="full_data_path", type=str, default="", help="Full path to take the dataset from"
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default="real_world_normals_bin_aug_2",
        help="The path to save the model",
    )
    parser.add_argument(
        "--model_finetune_path",
        dest="model_finetune_path",
        type=str,
        default=None,
        help="The path to load the starting model",
    )
    parser.add_argument(
        "--feas_model_path",
        dest="feasibility_model_path",
        type=str,
        default="Collision/panda_hfa_suction_gripper/2bins_real_rand_binary_aug_2",
        help="The path to feasibility model",
    )

    parser.add_argument(
        "--config_model_path",
        dest="config_model_path",
        type=str,
        default="~/amira_experiments",
        help="The path for configs and model",
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
        "--eval_freq", dest="evaluation_freq", type=int, default=100, help="The number of UPDATES to evalute state"
    )
    parser.add_argument(
        "--encoder",
        dest="encoder",
        type=str,
        default="Resnet43_8s",
        help="The pixel encoder architecture used in ConSACSSL",
    )
    parser.add_argument("--visualize", dest="visualize", action="store_true", help="If to visualize evaluation")
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
        "--ssl_lower_confidence", dest="ssl_lower_confidence", type=float, default=-1, help="The lower confidence threshold for pseudo label"
    )
    parser.add_argument(
        "--weight_decay", dest="weight_decay", type=float, default=1e-4, help="The weight decay for training"
    )
    parser.add_argument(
        "--recalc_reward",
        action="store_true",
        dest="recalc_reward",
        help="reclac the reward even if already calculated.",
    )
    parser.add_argument(
        "--pseudo_samples", dest="pseudo_samples", type=int, default=-1, help="The number of pseudo samples in each batch"
    )
    parser.add_argument("--eval_only",
                        action="store_true",
                        default=False,
                        help="only evaluate the model.")
    parser.add_argument("--use_pessimistic_loss",
                        action="store_true",
                        default=False,
                        help="use pessimistic_loss.")
    parser.add_argument("--train_only",
                        action="store_true",
                        default=False,
                        help="only train the model.")
    parser.add_argument("--save_best_eval_model",
                        action="store_true",
                        default=False,
                        help="save the best evaluation model.")
    parser.add_argument("--save_best_critic_model",
                        action="store_true",
                        default=False,
                        help="save the best train critic loss model.")
    parser.add_argument("--normalize_rewards",
                        action="store_true",
                        default=False,
                        help="normalize offline data rewards.")
    parser.add_argument("--ssl_method",
                        default="fixmatch",
                        help="select ssl method (default: fixmatch)",
                        choices=["online", "offline", "fixmatch","flexmatch", "contexture_flexmatch", "freematch", "contexture_freematch", "freematch_ent", "softmatch", "contexture_softmatch", "softmatchnew", "contexture_softmatchnew"])
    parser.add_argument("--weighted_method",
                        default=None,
                        help="0-1 weighted classes method",
                        choices=["softmax", "no_weighted"])
    parser.add_argument(
        "--note", dest="note", type=str, default="", help="note for tensorboard log & visualization"
    )
    parser.add_argument("--seed", dest="seed", type=int, default=-1, help="set random seed for reproducible")
    parser.add_argument(
        "--limit", dest="limit", type=int, default=None, help="online data memory buffer limit"
    )
    parser.add_argument("--ema_m", type=float, default=0.99, help="ema momentum for evaluating model")
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-4, help="learning rate")
        
    args = parser.parse_args()
    
    log_dir = os.path.expanduser(f"~/amira_experiments/logs/BinPicking/{args.log_dir}")
    feasibility_model_path = os.path.expanduser(
        f"~/amira_experiments/BinShifting-v1/feasibility_models/{args.feasibility_model_path}"
    )
    model_path = f"{args.config_model_path}/BinPicking-v1/{args.robot}/{args.gripper_config_file}/{args.model_path}"
    model_path = os.path.expanduser(model_path)
    
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    set_seed_everywhere(args.seed)

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
        weight_decay=args.weight_decay,
        optim="Adam",
        momentum=0.9,
        train_epochs=args.train_epochs,
        use_augmentation=True,
        use_pessimistic_loss=args.use_pessimistic_loss,
        weighted_method=args.weighted_method,
        sampler="random",
        # sampler_params=dict(pos_size=int(args.batch_size / 2)),  # use this if sampler="pos_neg"
        limit=args.limit,
        visualize=args.visualize,
        visualization_path=args.visualization_path,
        visualization_freq=args.visualization_freq,
        evaluation_freq=args.evaluation_freq,
        save_best_eval_model=args.save_best_eval_model,
        save_best_critic_model=args.save_best_critic_model,
        model_path=model_path,
        # for ssl learning (pseudo-labeling)
        w_offline=args.w_offline,
        w_online=args.w_online,
        w_pseudo=args.w_pseudo,
        ssl_confidence=args.ssl_confidence,
        ssl_lower_confidence=args.ssl_lower_confidence,
        ssl_method=args.ssl_method,
        ema_m=args.ema_m,
        pseudo_samples=args.pseudo_samples,
        note=args.note
    )

    if args.ssl_method == "flexmatch":
        agent = ConSACFlexMatch(params, log_path=log_dir)
    elif args.ssl_method == "contexture_flexmatch":
        agent = ConSACContextureFlexMatch(params, log_path=log_dir)  
    elif args.ssl_method == "freematch":
        agent = ConSACFreeMatch(params, log_path=log_dir)
    elif args.ssl_method == "contexture_freematch":
        agent = ConSACContextureFreeMatch(params, log_path=log_dir)
    elif args.ssl_method == "softmatch":
        agent = ConSACSoftMatch(params, log_path=log_dir)
    elif args.ssl_method == "contexture_softmatch":
        agent = ConSACContextureSoftMatch(params, log_path=log_dir)   
    elif args.ssl_method == "softmatchnew":
        agent = ConSACSoftMatchNew(params, log_path=log_dir)
    elif args.ssl_method == "contexture_softmatchnew":
        agent = ConSACContextureSoftMatchNew(params, log_path=log_dir)
    elif args.ssl_method == "online":
        agent = ConSACOnline(params, log_path=log_dir)
    elif args.ssl_method == "offline":
        agent = ConSACOffline(params, log_path=log_dir)
    else:
        agent = ConSACFixMatch(params, log_path=log_dir)
    if args.model_finetune_path:
        model_finetune_path = f"{args.config_model_path}/BinPicking-v1/{args.robot}/{args.gripper_config_file}/{args.model_finetune_path}"
        model_finetune_path = os.path.expanduser(model_finetune_path)
        agent.load_state(model_finetune_path)
    agent.init_summary_writer()
    agent.init_replay_memories()
    agent.load_state(model_path)
    # agent.load_feasibility_state(feasibility_model_path)
    gripper_meshes = load_meshes(os.path.expanduser(f"{args.config_model_path}/config/{args.gripper_config_file}.yaml"))
    bin_hull, bbox_center, bbox_hsize = load_bin(
        os.path.expanduser(f"{args.config_model_path}/config/{args.bin_config_file}.yaml")
    )
    # Tuebingen bin size: [0.6, 0.4, 0.28] (take halfsize) + 0.1 margin
    hm_size = 2 * bbox_hsize  # np.array([0.7, 0.5, 0.38])
    if args.full_data_path:
        data_path = os.path.expanduser(f"{args.full_data_path}")
    else:
        data_path = os.path.expanduser(f"~/amira_experiments/datasets/{args.bin_config_file}/{args.data_path}")
        if args.eval_path != "":
            eval_path = os.path.expanduser(f'~/amira_experiments/datasets/{args.bin_config_file}/{args.eval_path}')
            eval_paths = [
                os.path.join(eval_path, o) for o in os.listdir(eval_path) if os.path.isdir(os.path.join(eval_path, o)) and not o.startswith(".")
            ]

    data_paths = [
        os.path.join(data_path, o) for o in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, o)) and not o.startswith(".")
    ]

    if args.sparse_reward:
        memory_path_p = os.path.expanduser(
            f"~/amira_experiments/memories/{args.memory_path}"
        )
        agent.load_memory(memory_path_p, mark_synced=True)
        if args.eval_memory_path != "":
            eval_memory_path_p = os.path.expanduser(
                f"~/amira_experiments/memories/{args.eval_memory_path}"
            )
            agent.load_eval_memory(eval_memory_path_p)


    if args.eval_path != "":
        data_paths_train = data_paths[:72] # fit the batch
        data_paths_test = eval_paths
    else:
        train_cutoff = int(0.8 * len(data_paths))
        data_paths_train = data_paths[:train_cutoff]
        data_paths_test = data_paths[train_cutoff:]

    # This is needed if using python3.9 and OpenCV 4.5.5
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    if not os.path.exists(agent.tensorboard_logdir):
        os.makedirs(agent.tensorboard_logdir)
    
    # overwrite
    args.w_offline = agent.w_offline
    args.w_online = agent.w_online
    args.w_pseudo = agent.w_pseudo
    with open(os.path.join(agent.tensorboard_logdir, "args.json"), "w+") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.ssl_method == "offline":
        train_model = train_model_offline
    
    if args.eval_only:
        evaluate_model(agent=agent,
                   data_paths=data_paths_test,
                   bbox_center=bbox_center,
                   hm_size=hm_size,
                   hm_resolution=hm_resolution,
                   gripper_meshes=gripper_meshes,
                   visualize=args.visualize)
    elif args.train_only:
        train_model(
            agent=agent,
            train_epochs=args.train_epochs,
            model_path=model_path,
            data_paths=data_paths_train,
            hm_size=hm_size,
            bbox_center=bbox_center,
            hm_resolution=hm_resolution,
            gripper_meshes=gripper_meshes,
            batch_size=args.batch_size,
            recalc_reward=args.recalc_reward,
            debug=args.debug,
            normalize_rewards=args.normalize_rewards,
            freq_epoch=args.freq_epoch,
            w_offline=args.w_offline
        )
    else:
        train_model(
            agent=agent,
            train_epochs=args.train_epochs,
            model_path=model_path,
            data_paths=data_paths_train,
            hm_size=hm_size,
            bbox_center=bbox_center,
            hm_resolution=hm_resolution,
            gripper_meshes=gripper_meshes,
            batch_size=args.batch_size,
            recalc_reward=args.recalc_reward,
            debug=args.debug,
            normalize_rewards=args.normalize_rewards,
            freq_epoch=args.freq_epoch,
            w_offline=args.w_offline
        )

        evaluate_model(
            agent=agent,
            data_paths=data_paths_test,
            bbox_center=bbox_center,
            hm_size=hm_size,
            hm_resolution=hm_resolution,
            gripper_meshes=gripper_meshes,
            visualize=args.visualize
        )
