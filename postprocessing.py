import numpy as np 
import h5py
from datetime import datetime

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

def generate_depths(cam_params, points_3D):
    """Generate depth maps for each point across different camera views.
    
    Args:
        points_3D: A (# frames, # nodes, 3, # tracks) ndarray of 3D points in the world frame. 
        
        cam_params: (# frames, # cams, 9, # tracks) ndarray of camera parameters for each view, 
        organized similarly to bundle adjustment routine camera parameter inputs.
    
    Returns:
        depths: A (# frames, # nodes, # tracks, # cams) ndarray of depths for keypoints across separate views.
    """
    n_frames, n_cams, _, n_tracks = cam_params.shape 
    
    depths = []
    
    for cam in range(n_cams):
        view_depths = []
        
        for track in range(n_tracks):
            track_depths = []
            
            for frame in range(n_frames):
                points = points_3D[frame, ..., track]
                rot_vecs = cam_params[frame, [cam], :3, track]
                cam_depth = cam_params[frame, cam, 5, track]
                
                frame_depths = rotate(points, rot_vecs)
                frame_depths = frame_depths[:, -1] + cam_depth
                track_depths.append(frame_depths)
                
            track_depths = np.stack(track_depths, axis=0) # (frames, nodes)
            view_depths.append(track_depths)
            
        view_depths = np.stack(view_depths, axis=-1) # (frames, nodes, tracks)
        depths.append(view_depths)
    
    depths = np.stack(depths, axis=-1) # (frames, nodes, tracks, cams)
    return depths 

def reproj_views(cam_views, points):
    """Reproject across different views for a single track. 
    
    Args:
        cam_views: Multi camera parameter matrix as used in bundle adjustment (# cams, 9)
        
        points: A ndarray of shape (# samples, # nodes, 3)
    
    Returns:
        adjusted_reprojections: Array of length # cameras of (# samples, # nodes, 2) ndarrays of 2d points 
    """
    adjusted_reprojections = [] 
    n_cams = cam_views.shape[0]
    n_samples = points.shape[0]
    
    for cam in range(n_cams):
        reproj_view = []
        for sample in range(n_samples):
            proj = project(points[sample], cam_views[[cam], :])
            reproj_view.append(proj)
        reproj_view = np.concatenate([x[np.newaxis, ...] for x in reproj_view], axis=0) # (samples, nodes, 2)
        adjusted_reprojections.append(reproj_view)  
    
    return adjusted_reprojections

def reproj_tracks(optim_cam, optim_points):
    """Reproject across different views for multitrack point dataset. 
    
    Args:
        optim_cam: A (# frames, # cams, 9, # tracks) ndarray of optimized camera parameters for each frame / track.
                    9 refers to the number of parameters for each camera. The first 3 entries along that axis are 
                    rotation parameters, the next 3 are translation, then focal distance, then 2 distortion parameters. 
        
        optim_points: A (# frames, # nodes, 3, # tracks) ndarray of optimized 3D points in the world frame.
    
    Returns:
        reproj: A (# frames, # nodes, 2, # tracks, # cams) ndarray of optimized reprojections for each frame / track / view.
    """
    reproj = []
    
    n_frames, n_cams, _, n_tracks = optim_cam.shape    
    
    for track in range(n_tracks):
        reproj_track = []
    
        for frame in range(n_frames):
            cam_params = optim_cam[frame, ..., track]
            p3d = optim_points[[frame], ..., track] 
        
            reproj_frame = reproj_views(cam_params, p3d) # list of length n_cams of (1, nodes, 2)
            reproj_frame = np.stack(reproj_frame, axis=-1) # (1, nodes, 2, cams)
        
            reproj_track.append(reproj_frame) 
    
        reproj_track = np.concatenate(reproj_track, axis=0) # (frames, nodes, 2, cams)
        reproj.append(reproj_track)
    
    reproj = [np.stack((reproj[0][..., i], reproj[1][..., i]), axis=-1) for i in range(n_cams)]
    reproj = np.stack(reproj, axis=-1)
    
    return reproj



def save_optim_params(optim_p3d, optim_cam_params, optim_reproj, optim_depths, recording_name, file_name):
    """Store optimization results in a hdf5 file. 
    
    Args: 
        optim_p3d: A (# frames, # nodes, 3, # tracks) ndarray of optimized 3D points in the world frame.
        
        optim_cam_params: A (# frames, # cams, 9, # tracks) ndarray of optimized camera parameters for each frame / track.
                            9 refers to the number of parameters for each camera. The first 3 entries along that axis are 
                            rotation parameters, the next 3 are translation, then focal distance, then 2 distortion parameters. 
                            
        optim_reproj: A (# frames, # nodes, 2, # tracks, # cams) ndarray of optimized reprojections for each frame / track / view. 
                        Can be derived be projecting the optimized points into each camera view. 
                        
        optim_depths: A (# frames, # nodes, # tracks, # cams) ndarray of optimized reprojections for each frame / track / view. 
                        Can be derived by rotating the 3D points and then translating them for each camera view. 
                        
        recording_name: A string describing the recording being analyzed.
        
    Output: 
        file: A hdf5 file containing the following datasets:
                points: contains optim_p3d
                cam_params: contains optim_cam_params
                reprojections: contains optim_reproj
                depths: contains optim_depths 
                recording: contains recording name
                time: date time of creation 
    """
    points_attr = "A (# frames, # nodes, 3, # tracks) ndarray of optimized 3D points in the world frame."
    cam_attr = "A (# frames, # cams, 9, # tracks) ndarray of optimized camera parameters for each frame / track. 9 refers to the number of parameters for each camera. \
    The first 3 entries along that axis are rotation parameters, the next 3 are translation, then focal distance, then 2 distortion parameters."
    reproj_attr = "A (# frames, # nodes, 2, # tracks, # cams) ndarray of optimized reprojections for each frame / track / view. \
    Can be derived be projecting the optimized points into each camera view."
    depths_attr = "A (# frames, # nodes, # tracks, # cams) ndarray of optimized depths for each frame / track / view. \
    Can be derived by rotating the 3D points and then translating them for each camera view."
    
    file_name = file_name + '.analysis.h5'
    creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('points', data=optim_p3d, chunks=True, compression="gzip", compression_opts=1)
        f['points'].attrs['Description'] = points_attr
        
        f.create_dataset('cam_params', data=optim_cam_params, chunks=True, compression='gzip', compression_opts=1)
        f['cam_params'].attrs['Description'] = cam_attr
        
        f.create_dataset('reprojections', data=optim_reproj, chunks=True, compression="gzip", compression_opts=1)
        f['reprojections'].attrs['Descriptions'] = reproj_attr
        
        f.create_dataset('depths', data=optim_depths, chunks=True, compression="gzip", compression_opts=1)
        f['depths'].attrs['Description'] = depths_attr
        
        f.create_dataset('time created', data=creation_time)
        
        f.create_dataset('recording', data=recording_name)
                
def create_analysis_file(optim_p3d, optim_cam_params, recording_name, file_name):
    """Create post processing parameters and save all parameters in analysis file. 
    
    Args: 
        optim_p3d: A (# frames, # nodes, 3, # tracks) ndarray of optimized 3D points in the world frame.
        
        optim_cam_params: A (# frames, # cams, 9, # tracks) ndarray of optimized camera parameters for each frame / track.
                            9 refers to the number of parameters for each camera. The first 3 entries along that axis are 
                            rotation parameters, the next 3 are translation, then focal distance, then 2 distortion parameters. 
                        
        recording_name: A string describing the recording being analyzed.
        
    Output: 
        file: A hdf5 file containing the following datasets:
                points: contains optim_p3d
                cam_params: contains optim_cam_params
                reprojections: contains optim_reproj
                depths: contains optim_depths 
                recording: contains recording name
                time: date time of creation 
    """
    optim_depths = generate_depths(optim_cam_params, optim_p3d)
    optim_reproj = reproj_tracks(optim_cam_params, optim_p3d)
    save_optim_params(optim_p3d, optim_cam_params, optim_reproj, optim_depths, recording_name, file_name)
 