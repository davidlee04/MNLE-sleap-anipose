from enum import Enum
from typing import Callable
import numpy as np
from aniposelib.cameras import Camera, CameraGroup
import pandas as pd


# Multiview Linear Least Squares Triangulation
# Code courtesy of https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py


class TriangulationMethod(Enum):
    simple = 1
    calibrated_ransac = 2


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
    calibration_filepath: str = None,
    config: dict = {}
    ) -> np.ndarray:
    """Collect all 3D poses across all frames. 
    
    Args:
        poses_2d: A length # cameras list of pose matrices for a single animal. Each pose matrix is of 
        shape (# frames, # nodes, 2, # tracks).
        
        calibration_filepath: Filepath to calibration.toml.

        config: Dictionary with user parameters for triangulation.
        
    Returns:
        poses_3d: A (# frames, # nodes, 3, # tracks) that corresponds to the triangulated 3D points in the world frame. 
    """
    # Initializing the relevant looping variables and container variables
    n_cams, n_frames, n_nodes, _, n_tracks = poses_2d.shape
    poses_3d = []

    assert config is not {}, 'empty config'

    assert calibration_filepath is not None, 'calibration_filepath missing'

    cfg_tri = config['triangulation']

    ransac = cfg_tri['ransac']
    optim = cfg_tri['optim']
    show_progress = cfg_tri['progress']
    errors = []
    
    # Filling poses_3d with triangulated points
    for track in range(n_tracks):
        points = poses_2d[..., track]
        
        cgroup = CameraGroup.load(calibration_filepath)

        n_cams, n_frames, n_joints, _ = points.shape

        # DLT vs RANSAC
        if not ransac:
            points_shaped = points.reshape(n_cams, n_frames*n_joints, 2)
            cgroup = CameraGroup.load(calibration_filepath)
            points_3d = cgroup.triangulate(points_shaped, progress = show_progress)
            points_3d = points_3d.reshape((n_frames, n_joints, 3))
        else:
            # RANSAC has errors output
            n_cams, n_frames, n_joints, _ = points.shape
            points_shaped = points.reshape(n_cams, n_frames*n_joints, 2)
            points_3d, _, _, err = cgroup.triangulate_ransac(
                points_shaped,
                progress=show_progress)
            points_3d = points_3d.reshape((n_frames, n_joints, 3))
            err = err.reshape((n_frames, n_joints))
            errors.append(err)

        # optim parameter
        if optim:
            kwargs = {
                'scale_smooth': cfg_tri['scale_smooth'], 
                'scale_length': cfg_tri['scale_length'],
                'scale_length_weak': cfg_tri['scale_length_weak'],
                'reproj_error_threshold': cfg_tri['reproj_error_threshold'],
                'n_deriv_smooth': cfg_tri['n_deriv_smooth'],
                'constraints': cfg_tri['constraints'],
                'constraints_weak': cfg_tri['constraints_weak'],
                'verbose': cfg_tri['progress'],
                'reproj_loss': cfg_tri['reproj_loss']
            }
            points_3d = cgroup.optim_points(points, points_3d, **kwargs)

        poses_3d.append(points_3d)
    
    # Reshaping, putting track dimension at end
    poses_3d = np.stack(poses_3d, axis=-1)
    # Do the same for errors if ransac
    if ransac:
        errors = np.stack(errors, axis=-1)
    else:
        errors = np.array(errors)

    return poses_3d, errors

def load_constraints(config, bodyparts, key='constraints'):
    constraints_names = config['triangulation'].get(key, [])
    bp_index = dict(zip(bodyparts, range(len(bodyparts))))
    constraints = []
    for a, b in constraints_names:
        assert a in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(a)
        assert b in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(b)
        con = [bp_index[a], bp_index[b]]
        constraints.append(con)
    return constraints

def triangulate(config, calibration_file, poses_2d, scores_all, bodyparts):

    assert config is not {}, 'empty config'
    assert calibration_file is not None, 'calibration_filepath missing'

    config_tri = config['triangulation']
    ransac = config_tri['ransac']
    optim = config_tri['optim']
    show_progress = config_tri['progress']

    n_cams, n_frames, n_joints, _, n_tracks = poses_2d.shape

    poses_3d = []
    errors_3d = []
    points_proj = []
    scores_proj = []

    for track in range(n_tracks):
        points = poses_2d[..., track]
        scores = scores_all[..., track]

        bad = scores < config['triangulation']['score_threshold']
        points[bad] = np.nan
        
        cgroup = CameraGroup.load(calibration_file)

        if optim:
            constraints = load_constraints(config, bodyparts)
            constraints_weak = load_constraints(config, bodyparts, 'constraints_weak')

            points_2d = points
            scores_2d = scores

            points_shaped = points_2d.reshape(n_cams, n_frames*n_joints, 2)
            if ransac:
                points_3d_init, _, _, _ = cgroup.triangulate_ransac(
                    points_shaped,
                    progress=True)
            else:
                points_3d_init = cgroup.triangulate(points_shaped, progress=True)

            points_3d_init = points_3d_init.reshape(n_frames, n_joints, 3)

            c = np.isfinite(points_3d_init[:, :, 0])
            if np.sum(c) < 20:
                print("warning: not enough 3D points to run optimization")
            else:
                points_3d = cgroup.optim_points(
                    points_2d, points_3d_init,
                    constraints=constraints,
                    constraints_weak=constraints_weak,
                    # scores=scores_2d,
                    scale_smooth=config['triangulation']['scale_smooth'],
                    scale_length=config['triangulation']['scale_length'],
                    scale_length_weak=config['triangulation']['scale_length_weak'],
                    n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
                    reproj_error_threshold=config['triangulation']['reproj_error_threshold'],
                    verbose=True)

            points_2d_flat = points_2d.reshape(n_cams, -1, 2)
            points_3d_flat = points_3d.reshape(-1, 3)

            errors = cgroup.reprojection_error(points_3d_flat, points_2d_flat, mean=True)
            good_points = ~np.isnan(points[:,:,:,0])
            num_cams = np.sum(good_points, axis=0).astype('float')

            all_points_3d = points_3d_init
            all_errors = errors.reshape(n_frames, n_joints)

            scores[~good_points] = 2
            scores_3d = np.min(scores, axis=0)

            scores_3d[num_cams < 1] = np.nan
            all_errors[num_cams < 1] = np.nan

        else:
            points_2d = points.reshape(n_cams, n_frames*n_joints, 2)

            if ransac:
                points_3d, picked, p2ds, errors = cgroup.triangulate_ransac(
                    points_2d, min_cams=3, progress=True)

                all_points_picked = p2ds.reshape(n_cams, n_frames, n_joints, 2)
                good_points = ~np.isnan(all_points_picked[:, :, :, 0])

                num_cams = np.sum(np.sum(picked, axis=0), axis=1).reshape(n_frames, n_joints).astype('float')
            else:
                points_3d = cgroup.triangulate(points_2d, progress=True)
                errors = cgroup.reprojection_error(points_3d, points_2d, mean=True)
                good_points = ~np.isnan(points[:, :, :, 0])
                num_cams = np.sum(good_points, axis=0).astype('float')

            all_points_3d = points_3d.reshape(n_frames, n_joints, 3)
            all_errors = errors.reshape(n_frames, n_joints)

            scores[~good_points] = 2
            scores_3d = np.min(scores, axis=0)

            scores_3d[num_cams < 2] = np.nan
            all_errors[num_cams < 2] = np.nan
            num_cams[num_cams < 2] = np.nan

        M = np.identity(3)
        center = np.zeros(3)

        dout = pd.DataFrame()
        for bp_num, bp in enumerate(bodyparts):
            for ax_num, axis in enumerate(['x','y','z']):
                dout[bp + '_' + axis] = all_points_3d[:, bp_num, ax_num]
            dout[bp + '_error'] = all_errors[:, bp_num]
            dout[bp + '_ncams'] = num_cams[:, bp_num]
            dout[bp + '_score'] = scores_3d[:, bp_num]

        for i in range(3):
            for j in range(3):
                dout['M_{}{}'.format(i, j)] = M[i, j]

        for i in range(3):
            dout['center_{}'.format(i)] = center[i]

        dout['fnum'] = np.arange(n_frames)

        # dout.to_csv(output_fname, index=False)

        poses_3d.append(all_points_3d)
        errors_3d.append(all_errors)

        bp, points_2d_proj, scores_2d_proj = get_projected_points(config, dout, cgroup)
        print(f'scores_proj {scores_2d_proj.shape}')
        points_2d_proj = np.swapaxes(points_2d_proj, 1, 2)
        points_2d_proj = points_2d_proj.reshape(3, n_frames, n_joints, 2, 1)
        scores_2d_proj = np.swapaxes(scores_2d_proj, 0, 1)
        scores_2d_proj = scores_2d_proj.reshape(n_frames, n_joints, 1)
        points_proj.append(points_2d_proj)
        scores_proj.append(scores_2d_proj)
    
    # Reshaping, putting track dimension at end
    poses_3d = np.stack(poses_3d, axis=-1)
    errors_3d = np.stack(errors, axis=-1)
    points_proj = np.stack(points_proj, axis=-1)
    scores_proj = np.stack(scores_proj, axis=-1)

    return poses_3d, errors, points_proj, scores_proj

def get_projected_points(config, pose_data, cgroup):
    # pose_data = pd.read_csv(pose_fname)
    cols = [x for x in pose_data.columns if '_error' in x]
    bodyparts = [c.replace('_error', '') for c in cols]

    M = np.identity(3)
    center = np.zeros(3)
    for i in range(3):
        center[i] = np.mean(pose_data['center_{}'.format(i)])
        for j in range(3):
            M[i, j] = np.mean(pose_data['M_{}{}'.format(i, j)])

    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))

    all_points = np.array([np.array(pose_data.loc[:, (bp+'_x', bp+'_y', bp+'_z')])
                           for bp in bodyparts])

    all_errors = np.array([np.array(pose_data.loc[:, bp+'_error'])
                           for bp in bodyparts])

    all_scores = np.array([np.array(pose_data.loc[:, bp+'_score'])
                           for bp in bodyparts])
    
    if config['triangulation']['optim']:
        all_errors[np.isnan(all_errors)] = 0
    else:
        all_errors[np.isnan(all_errors)] = 10000
    good = (all_errors < 50)
    all_points[~good] = np.nan

    n_joints, n_frames, _ = all_points.shape
    n_cams = len(cgroup.cameras)

    all_points_flat = all_points.reshape(-1, 3)
    all_points_flat_t = (all_points_flat + center).dot(np.linalg.inv(M.T))

    points_2d_proj_flat = cgroup.project(all_points_flat_t)
    points_2d_proj = points_2d_proj_flat.reshape(n_cams, n_joints, n_frames, 2)

    return bodyparts, points_2d_proj, all_scores
