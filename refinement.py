import numpy as np 
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2 
from rich.progress import track as progress

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    
    Args:
        points: A (# points, 3) ndarray containing the points to be rotated. 
        
        rot_vecs: A (# points, 3) ndarray containing the rotation vectors corresponding to each point. 
        
    Returns:
        rotated_points: A (# points, 3) ndarray containing the rotated points. 
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def arrange_cam_params(rvecs, tvecs, focal_lengths, distortion_coeffs):
    """Arrange camera parameters as necessary for residual calculation and bundle adjustment. 
    
    Args:
        rvecs: A length # cams list of rotation vectors corresponding to each camera view 
        tvecs:  A length # cams list of translation vectors corresponding to each camera view 
        focal_lengths: A length # cams list of focal lengths corresponding to each camera view
        distortion_coeffs: A length # cams list of distortion coefficients corresponding to each camera view 
        
    Returns:
        ba_cam_params: A (# cameras, 9) size ndarray where the camera params are ordered per row. The first 3 columns 
        are the rotation vectors, the next 3 columns are translation vectors, the next column is the focal length, and the final 
        2 columns are the distortion coefficients. 
    """
    n_cams = len(rvecs) 
    ba_camera_params = np.full((n_cams, 9), np.nan)
    
    for i in range(n_cams):
        ba_camera_params[i, :3] = rvecs[i]
        ba_camera_params[i, 3:6] = tvecs[i]
        ba_camera_params[i, 6] = focal_lengths[i]
        ba_camera_params[i, 7:] = distortion_coeffs[i]
    
    return ba_camera_params

def initialize_cam_params(cameras_dict):
    """Initialize camera parameters from calibration file for bundle adjustment.
    
    Args:
        cameras_dict: A dictionary where each key corresponds to a camera view and the corresponding values are dictionaries 
        themselves. Also the output of preprocessing.get_camera_parameters(...). The value dictionaries are constructed as:
            'K': A (3,3) ndarray denoting the intrinsics matrix. 
            
            'R': A (3,3) ndarray denoting the rotation submatrix of the extrinsics matrix. 
            
            'T': A (3,) ndarray denoting the translation vector used to construct the extrinsics matrix. 
            
            'D': A (5,) ndarray denoting the distortion parameters. 
            
            'P': A (3,4) ndarray denoting the camera matrix calculated by P = K[R | T].
                    
    Returns:
        ba_cam_params: A (# cameras, 9) size ndarray where the camera params are ordered per row. The first 3 columns 
        are the rotation vectors, the next 3 columns are translation vectors, the next column is the focal length, and the final 
        2 columns are the distortion coefficients. 
    """
    rvecs = [camera['R'] for camera in cameras_dict.values()]
    rvecs = [cv2.Rodrigues(rvec)[0].flatten() for rvec in rvecs]
    tvecs = [camera['T'] for camera in cameras_dict.values()]
    distortions = [camera['D'][0][:2] for camera in cameras_dict.values()]
    focal_lengths = [(camera['K'][0,0] + camera['K'][1,1]) / 2 for camera in cameras_dict.values()]
    
    return arrange_cam_params(rvecs, tvecs, focal_lengths, distortions)

def initialize_point_params(p3d, p2d, sampling_rate = 1):
    """Initialize point parameters necessary for residual calculation and bundle adjustment. 
    
    Args:
        p3d: A (# frames, # nodes, 3) size ndarry consisting of initial estimates of 3D points.
        
        p2d: A # cams length list of (# frames, # nodes, 2) size ndarray consisting of 2D points across different views. 
        
        sampling rate: Sampling rate for frames, default is 1.
        
    Returns:
        points_3d: A (n_points, 3) size ndarray containing initial estimates of point coordinates in the world frame.
        
        camera_indices: A (n_observations,) size ndarray containing indices of cameras from 0 to n_cams - 1 involved in each observation.
        
        points_indices: A (n_observations,) size ndarray containing indices of points from 0 to n_points - 1 involved in each observation.
        
        points_2d: A (n_observations, 2) size ndarray containing measured 2-D coordinates of points projected on images in each observation.
    
    Note n_observations = # frames x # nodes x # cams = # cams x n_points
    """
    
    points_3d = []
    camera_indices = []
    point_indices = []
    points_2d = []
    
    n_points = 0 # counter of total number of 3d points 
    
    for cam in range(len(p2d)):
        # Pull out initial data 
        pts3d = p3d[::sampling_rate].reshape((-1, 3)) # (nodes X frames, 3)
        pts2d = p2d[cam][::sampling_rate].reshape((-1, 2)) # (nodes X frames, 2)
        
        n = len(pts3d)
        
        # Accumulate BA containers
        if cam == 0:
            points_3d.append(pts3d)
            pt_inds = np.arange(n) + n_points 
            n_points += n
            
        camera_indices.append(np.full((n,), cam))
        point_indices.append(pt_inds)
        points_2d.append(pts2d)
        
    points_3d = np.concatenate(points_3d, axis=0) # (nodes x frames, 3)
    camera_indices = np.concatenate(camera_indices, axis=0) # (nodes x views x frames)
    point_indices = np.concatenate(point_indices, axis=0)  # (nodes x views x franes)
    points_2d = np.concatenate(points_2d, axis=0) # (nodes x views x frames, 2)
    
    return points_3d, camera_indices, point_indices, points_2d

def iterative_bundle_adjustment(cameras_dict, tracks_3D, tracks_2D, sampling_rate = 1):
    """Perform iterative bundle adjustment for each track / frame. 
    
    Args:
        cameras_dict: A dictionary where each key corresponds to a camera view and the corresponding values are dictionaries 
        themselves. See initialize_cam_params for more detail. 
        
        tracks_3D: A (# frames, # nodes, 3, # tracks) ndarray containing the initial estimates for 3D points in the world frame.
        
        tracks_2D: A length # cams list of (# frames, # nodes, 2, # tracks) ndarrays containing the ground truth 2D points across views.
        
        sampling_rate: Sampling rate for frames, default is 1. 
    
    Returns:
        optim_cam: A (# frames, # cams, 9, # tracks) ndarray containing the camera parameters derived for each frame / track.
        
        optim_points: A (# frames, # nodes, 3, # tracks) ndarray containing the refined 3D points in the world frame. 
        
    """
    # Sampling from 2D and 3D points
    sampled_p3d = tracks_3D[::sampling_rate] # (frames, nodes, 3, tracks)
    sampled_p2d = np.stack([x[::sampling_rate] for x in tracks_2D], axis=-1) # (frames, nodes, 2, tracks, cams)
        
    # Grabbing relevant constants 
    n_frames, n_nodes, _, n_tracks, n_cams = sampled_p2d.shape
    
    # Initializing camera parameters /
    ba_camera_params = initialize_cam_params(cameras_dict)
    
    # Initializing container variables 
    optim_points = []
    optim_cam = []
    
    # Optimization loop
    for track in range(n_tracks):
        track_cam = []
        track_points = []
        
        for frame in progress(range(n_frames), description=f'Processing track {track}:'):
            p3d = sampled_p3d[[frame], ..., track]
            p2d = [sampled_p2d[[frame], ..., track, cam] for cam in range(n_cams)]
            
            # Initializing point parameters 
            points_3d, camera_indices, point_indices, points_2d = initialize_point_params(p3d, p2d, sampling_rate=sampling_rate)
            n_points = points_3d.shape[0]
            
            A = bundle_adjustment_sparsity(n_cams, n_points, camera_indices, point_indices)
            x0 = np.hstack((ba_camera_params.ravel(), points_3d.ravel()))
            
            # Optimization 
            res = least_squares(fun, x0, jac_sparsity=A, verbose=0, x_scale='jac', xtol=1e-4, method='trf', 
                           args=(n_cams, n_points, camera_indices, point_indices, points_2d))
            
            # Storing Results 
            cam_params = res.x[:(n_cams * 9)].reshape((n_cams, 9))
            points = res.x[(n_cams * 9):].reshape((n_nodes, 3))
            
            track_cam.append(cam_params)
            track_points.append(points)
            
        track_cam = np.stack(track_cam, axis=0) # (frames, nodes, cams, 9)
        track_points = np.stack(track_points, axis=0) # (frames, nodes, 3)
        
        optim_cam.append(track_cam)
        optim_points.append(track_points)
            
    optim_cam = np.stack(optim_cam, axis=-1) # (frames, cams, 9, tracks)
    optim_points = np.stack(optim_points, axis=-1) # (frames, nodes, 3, tracks)
    
    return optim_cam, optim_points 