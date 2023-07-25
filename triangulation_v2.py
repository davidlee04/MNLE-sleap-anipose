from enum import Enum
from typing import Callable
import numpy as np
from aniposelib.cameras import Camera, CameraGroup


# Multiview Linear Least Squares Triangulation
# Code courtesy of https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py


class TriangulationMethod(Enum):
    simple = 1
    calibrated_dtl = 2
    calibrated_ransac = 3


def triangulate_simple(points: np.ndarray, camera_mats: list) -> np.ndarray:
    """Triangulate the 3D positions of the points of interest using DLT algorithm.
    
    Args:
        points: A (# cameras, 2) ndarray containing the (x, y) coordinates of the point of interest for each camera view
        camera_mats: A length # cameras list containing the (3,4) ndarrays which are the camera matrices for each camera. 
        Note that the order of the camera matrices has to match with the ordering of the points. 
        
    Returns: 
        poses_3d: A (3,) ndarray corresponding to the triangulated 3D vector.  Computation is done via the DLT algorithm, see here for more: 
        http://bardsley.org.uk/wp-content/uploads/2007/02/3d-reconstruction-using-the-direct-linear-transform.pdf
    """
    # Initializing the coefficients matrix for the least squares problem
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    
    # Filling in the coefficients matrix
    for i, mat in enumerate(camera_mats):
        x, y = points[i]

        # Adding the entries to the coefficient matrix for the particular camera view
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0] 
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
        
    # Solving the linear least squares problem to grab the homogeneous 3D coordinates
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    poses_3d = vh[-1]

    # Converting to inhomogeneous coordinates
    poses_3d = poses_3d[:3] / poses_3d[3]
    
    return poses_3d


def get_3d_poses(
    poses_2d: list,
    camera_mats: list = [],
    calibration_filepath: str = None,
    triangulate: TriangulationMethod = TriangulationMethod.simple,
    refine_calibration: bool = False,
    show_progress: bool = False
    ) -> np.ndarray:
    """Collect all 3D poses across all frames. 
    
    Args:
        poses_2d: A length # cameras list of pose matrices for a single animal. Each pose matrix is of 
        shape (# frames, # nodes, 2, # tracks).
        
        camera_mats: A length # cameras list of camera matrices. Each camera matrix is a (3,4) ndarray. 
        Note that the camera matrices and pose_2d matrices have to be ordered in a corresponding fashion.
        or
        calibration_filepath: Filepath to calibration.toml

        triangulate: Triangulation method
            - simple: No other options are required
            - calibrated_dtl: refine_calibration, show_progress can be passed
            - calibrated_ransac: refine_calibration, show_progress can be passed

        refine_calibration: bool = False, Use CameraGroup.optim refinement

        show_progress: bool = False, Show progress of calibration
        
    Returns:
        poses_3d: A (# frames, # nodes, 3, # tracks) that corresponds to the triangulated 3D points in the world frame. 
    """
    # Initializing the relevant looping variables and container variables
    n_cams, n_frames, n_nodes, _, n_tracks = poses_2d.shape
    poses_3d = []
    
    # Filling poses_3d with triangulated points
    for track in range(n_tracks):
        points = poses_2d[..., track]
        if triangulate == TriangulationMethod.simple:
            n_cams, n_frames, n_joints, _ = points.shape
              # constraints = np.array(constraints)
              # constraints_weak = np.array(constraints_weak)
            
            points_shaped = points.reshape(n_cams, n_frames*n_joints, 2)
            cgroup = CameraGroup.load(calibration_filepath)
            points_3d = cgroup.triangulate(points_shaped, progress = show_progress)
            points_3d = points_3d.reshape((n_frames, n_joints, 3))
            
        else:
            assert calibration_filepath is not None, 'calibration_filepath missing'
            cgroup = CameraGroup.load(calibration_filepath)
            if triangulate == TriangulationMethod.calibrated_dtl:
                init_ransac = False
            elif triangulate == TriangulationMethod.calibrated_ransac:
                init_ransac = True
                # refine_calibration = False # applies only to DLT
            else:
                error('Incorrect triangulation type')
            # points_3d = cgroup.triangulate_optim(
            #     poses_2d[..., track],
            #     init_ransac=init_ransac,
            #     init_progress=show_progress)
            n_cams, n_frames, n_joints, _ = points.shape
            if init_ransac:
                # calibrated_ransac
                points_shaped = points.reshape(n_cams, n_frames*n_joints, 2)
                points_3d, *_ = cgroup.triangulate_ransac(
                    points_shaped,
                    progress=show_progress)
                points_3d = points_3d.reshape((n_frames, n_joints, 3))
            else:
                # calibrated_dtl
                points_3d, *_ = cgroup.triangulate_possible(points, progress=show_progress)
                points_3d = points_3d.reshape((n_frames, n_joints, 3))

        if refine_calibration:
            # constraints?
            points_3d = cgroup.optim_points(points, points_3d)

        poses_3d.append(points_3d)
        
    poses_3d = np.stack(poses_3d, axis=-1)

    return poses_3d
