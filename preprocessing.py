import numpy as np 
import h5py 
import json
from scipy.interpolate import interp1d


def get_cam_matrix(K: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Calculate the camera matrix given the intrinsic and extrinsic matrices.
    
    Args:
        K: A (3,3) ndarray denoting the intrinsics matrix. 
        
        R: A (3,3) ndarray denoting the rotation submatrix of the extrinsics matrix.
        
        T: A (3,) ndarray denoting the translation vector which is the final column of the extrinsics matrix. 
        
    Returns:
        P: A (3,4) ndarray, the camera matrix P which is calculated as P = K[R | T].
    """
    extrinsics = np.concatenate((R, T[:, np.newaxis]), axis=1)
    return K @ extrinsics

def get_camera_params(calib_file: str, show_world_frame: bool = True) -> dict:
    """Read a multical calibration file and return a dictionary containing the camera parameters for all cameras.
    
    Args:
        calib_file: A string of the path to the json calibration file that is derived from multical calibration.
        
        show_world_frame: A flag determining whether or not the user wants to explicitly see which camera view is the 
        world reference frame. Multical usually sets the first folder of images it finds as the world reference frame, 
        however, the user can also select the world reference frame. Setting the flag to be true prints out the 
        camera corresponding to the world frame.
        
    Returns:
        cam_params: A dictionary where each key corresponds to a camera view and the corresponding values are dictionaries 
        themselves. The value dictionaries are constructed as:
            'K': A (3,3) ndarray denoting the intrinsics matrix. 
            
            'R': A (3,3) ndarray denoting the rotation submatrix of the extrinsics matrix. 
            
            'T': A (3,) ndarray denoting the translation vector used to construct the extrinsics matrix. 
            
            'D': A (5,) ndarray denoting the distortion parameters. 
            
            'P': A (3,4) ndarray denoting the camera matrix calculated by P = K[R | T].
    """
    # Reading the data from the json file 
    with open(calib_file, 'r') as f:
        data = json.load(f)
    
    # Separating the extrinsics and intrinsics
    cams = data['cameras']
    cam_poses = data['camera_poses']
    
    cam_ids = list(cams.keys()) # List of camera views , i.e. camera_1
    poses = list(cam_poses.keys()) # List of camera poses, i.e. camera_1_to_world_frame
    
    # Initializing and filling the camera parameters dictionary
    cam_params = {}
    
    for cam, pose in zip(cam_ids, poses):
        intrinsics = cams[cam]
        extrinsics = cam_poses[pose]
        
        K = np.array(intrinsics['K'])
        R = np.array(extrinsics['R'])
        T = np.array(extrinsics['T'])
        D = np.array(intrinsics['dist'])
        P = get_cam_matrix(K, R, T)
        
        cam_params[cam] = {
            'K': K,
            'R': R,
            'T': T,
            'D': D,
            'P': P
        }
        
    if show_world_frame:
        world_frame = poses[np.argmin([len(pose) for pose in poses])]
        print(f'The world reference frame is {world_frame}')
    
    return cam_params

def fill_missing(Y: np.ndarray, kind: str = 'linear') -> np.ndarray:
    """Fill missing values independently along each dimension after the first.
    
    Args:
        Y: A ndarray of arbitrary shape to be cleaned by removing nan values and interpolating along each dimension after the first.
        
        kind: A string denoting the kind of interpolation to do. For full details refer to scipy.interpolate.interp1d().
        
    Returns:
        Y: A ndarray of the same shape as Y.
    """
    # Store initial shape
    initial_shape = Y.shape
    
    # Flatten after fist dim 
    Y = Y.reshape((initial_shape[0], -1))
    
    # Interpolate along each slice
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        
        # Build interpolant
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
        
        # Fill missing 
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing nans with the nearest non-nan values 
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        
        # Save slice 
        Y[:, i] = y
        
    # Restore to initial shape 
    Y = Y.reshape(initial_shape)
    
    return Y

def get_2d_poses(analysis_file: str, show_nodes: bool = False, show_file: bool = False, show_dsets: bool = False, clean: bool = False) -> np.ndarray:
    """Retrive the 2D poses of the animals given the hdf5 analysis file.
    
    Args:
        analysis_file: A string containing the path to the hdf5 analysis file derived from SLEAP.
        
        clean: A flag determining whether or not the user wants the 2d poses to be cleaned of nans using the interp_1d function. 
        
    Returns:
        poses_2d: A (# frames, # nodes, 2, # tracks) ndarray denoting the 2d locations of the nodes for each animal across all analyzed frames.
    """
    # Display file being read
    if show_file:
            print(f'File location: {analysis_file}')
            print()
    
    # Read file
    with h5py.File(analysis_file, 'r') as f:
        # Display the datasets in the hdf5 file
        if show_dsets:
            print(f'Datasets in hdf5 file are: {f.keys()}')
            print()
            
        # Display nodes and their indices
        if show_nodes:
            nodes = [n.decode() for n in f['node_names'][:]]
            for i, node in enumerate(nodes):
                print(f'{i}: {node}')
                print()
    
        # Grab poses and print shape, which should be (# frames, # nodes, 2, # tracks)
        poses_2d = f['tracks'][:].T
    
    # Interpolate the nan values in the tracks        
    if clean:
        poses_2d = fill_missing(poses_2d, kind='linear')
        
    return poses_2d

def load_cams_poses(analysis_files, calibration_file, show_world_frame = True, show_nodes: bool = False, show_file: bool = True, show_dsets: bool = False, clean: bool = False):
    """Load the 2D tracks across all views and the corresponding camera parameters. 
    
    Args:
        analysis_files: A length # cams list of the different analysis.h5 files corresponding to different camera views. 
        
        calibration_file: A string corresponding to the json with the camera parameters (output of multical calibration).
        
    Returns:
        cam_params: A dictionary where each key corresponds to a camera view and the corresponding values are dictionaries 
        themselves. The value dictionaries are constructed as:
            'K': A (3,3) ndarray denoting the intrinsics matrix. 
            
            'R': A (3,3) ndarray denoting the rotation submatrix of the extrinsics matrix. 
            
            'T': A (3,) ndarray denoting the translation vector used to construct the extrinsics matrix. 
            
            'D': A (5,) ndarray denoting the distortion parameters. 
            
            'P': A (3,4) ndarray denoting the camera matrix calculated by P = K[R | T].
            
        tracks_2D: A length # cams list of (# frames, # nodes, 2, # tracks) ndarrays denoting the 2D poses of each animal across all frames. 
    """
    cam_params = get_camera_params(calibration_file, show_world_frame=show_world_frame)
    tracks_2D = [get_2d_poses(analysis_file, show_nodes=show_nodes, show_file=show_file, show_dsets=show_dsets, clean=clean) for analysis_file in analysis_files]
    
    return cam_params, tracks_2D

def get_point_scores(analysis_file):
    with h5py.File(analysis_file, 'r') as f:
        point_scores = f['point_scores'][:].T

    # should be # frames x # bodyparts x # tracks
    return point_scores

def get_bodyparts(analysis_file):
    with h5py.File(analysis_file, 'r') as f:
        bodyparts = f['node_names'][:]
        bodyparts = [bp.decode('UTF-8') for bp in bodyparts]

    return bodyparts