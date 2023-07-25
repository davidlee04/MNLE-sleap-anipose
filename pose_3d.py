# Generic imports
from os import listdir
from os.path import join, split, exists, dirname
import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import re
from scipy.io import savemat
import importlib
import h5py
import shutil
from IPython.utils import io
import toml

# Custom libraries 
spec = importlib.util.find_spec('aniposelib')
if spec is None:
	print("aniposelib is not installed")
#     !python -m pip install --user aniposelib
from aniposelib.cameras import CameraGroup
from preprocessing import get_2d_poses, get_point_scores
from aniposelib.utils import load_pose2d_fnames
from triangulation_v3 import TriangulationMethod, get_3d_poses
from filter import get_filter_poses
from calibration import generate_calibration
from convert import convert_slp_to_h5

from os import mkdir, rename

config = dict()
cam_names = []

DEFAULT_CONFIG = {
    'filter': {
    	'type': 'median',
    	'medfilt': 15,
    	'offset_threshold': 10,
    	'spline': False,
        'score_threshold': 0.05,
        'multiprocessing': False,
        'n_back': 4,
    },
    
    'calibration': {
        'board_type': 'charuco',
        'board_size': [5,5],
        'board_marker_bits': 4,
        'board_marker_dict_number': 100,
        'board_marker_length': 3.75,
        #'board_marker_separation_length': 1, # only for aruco
        'board_square_side_length': 5,
        'fisheye': False,
        'video_extension': 'mp4',
        'n_cams': 3,
        # 'cam_naming': 'letter',
        # 'cam_regex': 'Cam([A-Z])',
        #'Cam([A-Z])'
        
    },
    
    'triangulation': {
        'ransac': False,
        'optim': True,
        'progress': True,
        'constraints': [],
        'constraints_weak': [],
        'scale_smooth': 4,
        'scale_length': 2,
        'scale_length_weak': 0.5,
        'reproj_error_threshold': 15,
        'n_deriv_smooth': 1,
        'reproj_loss': 'soft_l1'
    },
}

def process_config(project_directory):
	global config

	if not exists(project_directory):
		print('invalid directory') # to change
		return

	if not exists(join(project_directory, 'config.toml')):
		print('config.toml not found')
		return

	config_file_path = join(project_directory, 'config.toml')

	try:
		loaded_config = toml.load(config_file_path)
	except(TypeError, toml.TomlDecodeError):
		print('error loading config.toml') # to change
		return

	if 'path' not in loaded_config:
		print('session path not specified') # to change
		return

	"""Loads config object (dict).

	Args:
		passed_config: Dictionary representing config file
	"""
	
	config = loaded_config

	for key, value in DEFAULT_CONFIG.items():
		if key not in config:
			config[key] = value
		elif isinstance(value, dict):
			for key2, value2 in value.items():
				if key2 not in config[key]:
					config[key][key2] = value2

	process_files()

def process_files():
	"""File preprocessing."""
	global cam_names
	print(config['path'])
	# Generate file paths

	# Generate list of .h5 files in session_filepath
	
	print('Read files from session path')

	calibration_folder = config['path']+'/../calibration'
	calibration_toml_path = join(calibration_folder, 'calibration.toml')

	if exists(calibration_toml_path):
		try:
			calibration_toml = toml.load(calibration_toml_path)
			cam_names = []
			for key, value in calibration_toml.items():
				if key == 'metadata':
					continue
				cam_name = calibration_toml[key]['name']
				if cam_name.lower().find('cam')==-1:
					cam_name = f'Cam{cam_name}'
				cam_names.append(cam_name)
		except(TypeError, toml.TomlDecodeError):
			print('error loading calibration.toml file') # to change
			return

def convert():
	convert_slp_to_h5(config)

def calibrate():
	global cam_names
	path = config['path']
	calibration_folder = path+'/../calibration'
	cal_toml_file = join(calibration_folder, 'calibration.toml')

	cam_names_regex_patterns = {
		'letter': r'Cam([A-Za-z])',
		'number': r'Cam([\d+])'
	}

	if 'cam_naming' in config['calibration']:
		config['calibration']['cam_regex'] = cam_names_regex_patterns[config['calibration']['cam_naming']]
	elif 'cam_regex' not in config['calibration']:
		print('error, can\'t find cam_regex or cam_naming field')
		return
		
	generate_calibration(config)
	

	assert exists(cal_toml_file), 'error, calibration.toml not found'
	try:
		cal_toml = toml.load(cal_toml_file)
		cam_names = []
		for key, value in cal_toml.items():
			if key == 'metadata':
				continue
			cam_names.append(cal_toml[key]['name'])
	except(TypeError, toml.TomlDecodeError):
		print('error loading calibration.toml file') # to change

def filter_poses():
	predictions_h5_directory = config['path']+'/predictions_h5'
	# h5_regex = re.compile('.*Cam.*.h5', re.IGNORECASE)
	# file_list = list(filter(h5_regex.match, listdir(predictions_h5_directory)))

	camera_ids = cam_names
	print(f'CAMERA NAMES {camera_ids}')

	cam1_regex = re.compile(f'.*{camera_ids[0]}.*.h5')
	cam1_file_list = list(filter(cam1_regex.match, listdir(predictions_h5_directory)))

	# Loop through 'Cam0' files
	for cam1_file in cam1_file_list:

	    # Get files for camera group (cam1,2,3) for a given trial
	    camera_group = [cam1_file.replace(camera_ids[0], cname) for cname in camera_ids]

	    """Get 2D data."""
	    # The get_2D_poses function returns 2D tracks for a single file at a time,
	    # so we append all the tracks to a list and then stack the 2D tracks on top of each other.
	    print(f'Processing {camera_group}')
	    for cam_file in camera_group:
	    	pose2d = get_2d_poses(join(predictions_h5_directory, cam_file))
	    	
	    	n_frames, n_nodes, _, n_tracks = pose2d.shape
	    	# p2d = np.copy(pose2d).reshape(n_frames, n_nodes, 1, 2)
	    	pose2d = pose2d.reshape(n_frames, n_nodes, 1, 2)

	    	point_scores = get_point_scores(join(predictions_h5_directory, cam_file))
	    	
	    	# point_scores = point_scores.reshape(n_frames, n_nodes, 1)

	    	filtered_points, filtered_scores = get_filter_poses(config, pose2d, point_scores)
	    	filtered_scores = filtered_scores.reshape(n_frames, n_nodes, 1)

	    	filtered_points = filtered_points.reshape(n_frames, n_nodes, 2, 1)
	    	# split on .
	    	ind = cam_file.find('.h5')
	    	assert ind != -1, 'not an h5!'
	    	out_file = cam_file[0:ind]+'_filtered'+cam_file[ind:]
	    	save_as_h5(predictions_h5_directory, cam_file, out_file, filtered_points, filtered_scores)

def save_as_h5(original_directory, original_file, filter_file, points, scores):
	filter_directory = join(config['path'],'predictions_filtered_h5')
	if not exists(filter_directory):
		mkdir(filter_directory)
	shutil.copyfile(join(original_directory, original_file), join(filter_directory, filter_file))
	with h5py.File(join(filter_directory, filter_file), 'r+') as f:
		points = points.T
		scores = scores.T
		h5_points = f['tracks']
		h5_scores = f['point_scores']
		h5_points[...] = points
		h5_scores[...] = scores

def triangulate_poses():
	"""Goes through cameras (as .h5 files) and triangulates 3D positions.

	Args:
		calibration_file: String specifying file path of calibration.toml
		session_filepath: String specifying file path of camera prediction files
		session_name: String specifying folder name of camera prediction files
		save_filepath: String specifying file path of output matrices
		file_list: List of camera prediction files
	"""
	camera_ids = cam_names
	assert config is not {}, 'empty config'
	session_directory = join(config['path'], 'predictions_h5')
	# h5_regex = re.compile('.*Cam.*.h5', re.IGNORECASE)
	# file_list = list(filter(h5_regex.match, listdir(session_filepath)))
	if exists(join(config['path'], 'predictions_filtered_h5')):
		session_directory = join(config['path'], 'predictions_filtered_h5')

	print(f'SESSION DIRECTORY {session_directory}')

	print(f'CAMERA NAMES {camera_ids}')

	calibration_file = config['path']+'/../calibration/calibration.toml'
	print(f'CALIBRATION FILE {calibration_file}')

	_, session_name = split(session_directory)
	print(f'SESSION_NAME {session_name}')

	save_filepath = join(config['path'], 'pose3d')

	# Make directory if doesn't exist
	if not exists(save_filepath):
		mkdir(save_filepath)

	# Generate list of 'Cam0' files (cam1_file_list)

	cam1_regex = re.compile(f'.*{camera_ids[0]}.*.h5')
	cam1_file_list = list(filter(cam1_regex.match, listdir(session_directory)))

	# Loop through 'Cam0' files
	for cam1_file in cam1_file_list:
	    # Get files for camera group (cam1,2,3) for a given trial
	    cam_group = [cam1_file.replace(camera_ids[0], cname) for cname in camera_ids]

	    print(f'Processing {cam_group}')

	    """Get trial data from h5 file."""
	    with h5py.File(join(session_directory, cam1_file), 'r') as cam1_h5_file:
	        marker_names = list(cam1_h5_file['node_names'][()])

	    video_files = []
	    for cam_file in cam_group:
	        with h5py.File(join(session_directory, cam_file), 'r') as cam_h5_file:
	            video_files.append(cam_h5_file['video_path'][()])

	    """Get 2D data."""
	    # The get_2D_poses function returns 2D tracks for a single file at a time,
	    # so we append all the tracks to a list and then stack the 2D tracks on top of each other. 
	    pose2d = [get_2d_poses(join(session_directory, cam_file)) for cam_file in cam_group]

	    # ? unsure... just converting to nparray
	    pose2d = np.stack(pose2d, axis=0)[:, :]
	    
	    n_cams, n_frames, n_nodes, _, n_tracks = pose2d.shape

	    # Thresholding with score_threshold
	    
	    # get point_scores from .h5
	    point_scores = [get_point_scores(join(session_directory, cam_file)) for cam_file in cam_group]
	    # stack cameras on top of each other 3 x tracks x nodes x 1 (already technically this dimension... turn to nparray)
	    # matching line 277
	    point_scores = np.stack(point_scores, axis=0)[:, :]

	    point_scores = point_scores.reshape(n_cams, n_frames, n_nodes)

	    # get indices where point_scores < score_threshold
	    # filtered = point_scores.reshape(n_cams*n_frames*n_nodes*2*n_tracks)
	    bad = point_scores < config['triangulation']['score_threshold']

	    # turn corresponding indices in pose2d to np.nan
	    pose2d[bad] = np.nan
	    print(f'POSE2d SHAPE {pose2d.shape}')

	    """Get 3D data from triangulation."""
	    """Aniposelib gives us the option to triangulate with the direct linear transformation (DLT) or with RANSAC,
	    which adds an outlier rejection subroutine to the DLT. <br>
	    In addition to these 2 triangulation methods, we can further refine the 3D points via direct optimization
	    of the reprojection error.
	    def get_3d_poses(
	                    poses_2d: list,
	                    camera_mats: list = [],
	                    calibration_filepath: str = None,
	                    triangulate: TriangulationMethod = TriangulationMethod.simple,
	                    refine_calibration: bool = False,
	                    show_progress: bool = False
	                    ) -> np.ndarray
	    Args:
	        poses_2d: A length # cameras list of pose matrices for a single animal. Each pose matrix is of 
	        shape (# frames, # nodes, 2, # tracks).
	        
	        camera_mats: A length # cameras list of camera matrices. Each camera matrix is a (3,4) ndarray. 
	        Note that the camera matrices and pose_2d matrices have to be ordered in a corresponding fashion.
	        or
	        calibration_filepath: Filepath to calibration.toml

	        triangulate: Triangulation method
	            - simple
	            - calibrated_dtl
	            - calibrated_ransac

	        refine_calibration: bool = False, Use CameraGroup.optim refinement

	        show_progress: bool = False, Show progress of calibration
	        
	    Returns:
	        poses_3d: A (# frames, # nodes, 3, # tracks) that corresponds to the triangulated 3D points in the world frame. 
	    """
	    
	    #with io.capture_output() as captured:
	    
	    pose3d, errors = get_3d_poses(poses_2d=pose2d,
	                    calibration_filepath=calibration_file,
	                    config=config)

	    print(errors.shape)

	    # Save as .mat to save_filepath
	    trial_name = cam1_file
	    save_as_mat(session_name, trial_name, marker_names, video_files, pose3d, save_filepath, errors)

def save_as_mat(session_name, trial_name, marker_names, video_files, pose3d, save_filepath, errors):
	"""Save pose3d data to .mat file in save_filepath.
	"""
	mat_dict = {'session': session_name,
	            'trial': trial_name.replace('.h5', ''),
	            'marker_names': marker_names,
	            'video_files': video_files,
	            'pose3d': pose3d,
	            'errors': errors}
	mat_file = join(save_filepath, trial_name.replace('.h5', '.mat'))

	print(f'Saving pose 3d to {mat_file}')
	savemat(mat_file, mat_dict)