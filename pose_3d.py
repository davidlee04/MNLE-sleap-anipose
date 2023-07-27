# Generic imports

import numpy as np
import h5py
import re
import importlib
import h5py
import shutil
import toml

from os import listdir, mkdir, rename
from os.path import join, split, exists, dirname

from scipy.io import savemat

# Custom libraries 
spec = importlib.util.find_spec('aniposelib')
if spec is None:
	print("aniposelib is not installed")
#     !python -m pip install --user aniposelib

from preprocessing import get_2d_poses, get_point_scores
from triangulation_v3 import get_3d_poses
from filter import get_filter_poses
from calibration import generate_calibration
from convert import convert_slp_to_h5

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
	"""
	Loads config.toml from project_directory

	Args:
		project_directory: file path of directory containing config.toml
	"""
	global config

	if not exists(project_directory):
		print('invalid directory')
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

	# Reject config.toml without a 'path' key
	if 'path' not in loaded_config:
		print('session path not specified') # to change
		return

	config = loaded_config

	# Load default values from DEFAULT_CONFIG for omitted values in config.toml
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

	if not exists(join(config['path'],'predictions')):
		print('error, predictions directory does not exist')
		return

	calibration_folder = config['path']+'/../calibration'
	calibration_toml_path = join(calibration_folder, 'calibration.toml')

	# During file preprocessing, attempt to load cam_names from calibration.toml if exists
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
	"""Calls convert function with config parameters."""
	convert_slp_to_h5(config)

def calibrate():
	"""
	Creates calibration.toml according to calibration videos.

	There should exist a calibration.toml by the end of this function.
	Also loads cam_names from calibration.toml.
	"""

	global cam_names
	path = config['path']
	calibration_folder = path+'/../calibration'
	cal_toml_file = join(calibration_folder, 'calibration.toml')

	cam_names_regex_patterns = {
		'letter': r'Cam[A-Za-z]',
		'number': r'Cam[\d+]'
	}

	# Choosing regex pattern based on user inputs from config.toml
	if 'cam_naming' in config['calibration']:
		config['calibration']['cam_regex'] = cam_names_regex_patterns[config['calibration']['cam_naming']]
	elif 'cam_regex' not in config['calibration']:
		print('error, can\'t find cam_regex or cam_naming field')
		return

	# calibration.toml creation
	generate_calibration(config)

	# Sanity check
	assert exists(cal_toml_file), 'error, calibration.toml not found'

	# Loads cam_names from calibration.toml file
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
	"""Generate filtered .h5 files by applying 2D filters."""

	predictions_h5_directory = config['path']+'/predictions_h5'

	camera_ids = cam_names
	print(f'CAMERA NAMES {camera_ids}')

	# Get all .h5 files in predictions_h5 that have our first camera name
	# e.g. asdf_Cam0_1234.h5
	cam1_regex = re.compile(f'.*{camera_ids[0]}.*.h5')
	cam1_file_list = list(filter(cam1_regex.match, listdir(predictions_h5_directory)))

	# Separately analyze each trial
	for cam1_file in cam1_file_list:

	    # Each video from each camera for a given trial
	    # e.g. [asdf_Cam0_1234.h5, asdf_Cam1_1234.h5, asdf_Cam2_1234.h5] 
	    camera_group = [cam1_file.replace(camera_ids[0], cname) for cname in camera_ids]

	    """Get 2D data."""
	    # The get_2D_poses function returns 2D tracks for a single file at a time,
	    # so we append all the tracks to a list and then stack the 2D tracks on top of each other.
	    print(f'Processing {camera_group}')
	    for cam_file in camera_group:
	    	pose2d = get_2d_poses(join(predictions_h5_directory, cam_file))
	    	
	    	n_frames, n_nodes, _, n_tracks = pose2d.shape
	    	
	    	# Reshaping for get_filter_poses
	    	pose2d = pose2d.reshape(n_frames, n_nodes, 1, 2)

	    	# Prediction scores for each point
	    	point_scores = get_point_scores(join(predictions_h5_directory, cam_file))

	    	filtered_points, filtered_scores = get_filter_poses(config, pose2d, point_scores)

	    	# Reshaping for exporting back to .h5
	    	filtered_points = filtered_points.reshape(n_frames, n_nodes, 2, 1)
	    	filtered_scores = filtered_scores.reshape(n_frames, n_nodes, 1)
	    	
	    	# Append '_filtered' to end of file name
	    	ind = cam_file.find('.h5')
	    	# Sanity check
	    	assert ind != -1, 'not an h5!'
	    	out_file = cam_file[0:ind]+'_filtered'+cam_file[ind:]
	    	to_filtered_h5(predictions_h5_directory, cam_file, out_file, filtered_points, filtered_scores)

def to_filtered_h5(original_directory, original_file, filter_file, points, scores):
	"""
	Generates new .h5 file with filtered points and scores.

	Copies original .h5 in original_directory/original_file to filter_file
	and updates it with filtered points and scores.
	Helper function to filter_poses().

	Args:
		original_directory: directory containing original_file
		original_file: original .h5 file
		filter_file: new filtered .h5 file
		points: filtered points
		scores: filtered scores
	"""

	# Save to 'predictions_filtered_h5' directory
	filter_directory = join(config['path'],'predictions_filtered_h5')
	if not exists(filter_directory):
		mkdir(filter_directory)

	# Copying original .h5 to new location
	shutil.copyfile(join(original_directory, original_file), join(filter_directory, filter_file))

	# Overwrite new .h5 with updated filtered points/scores
	with h5py.File(join(filter_directory, filter_file), 'r+') as f:
		points = points.T
		scores = scores.T
		h5_points = f['tracks']
		h5_scores = f['point_scores']
		h5_points[...] = points
		h5_scores[...] = scores

def triangulate_poses():
	"""Triangulates 3D poses as .mat files from .h5 files."""

	camera_ids = cam_names

	assert config is not {}, 'empty config'

	# Run analysis on filtered data if directory exists, otherwise on raw data
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

	# Output to pose3d directory
	save_filepath = join(config['path'], 'pose3d')
	if not exists(save_filepath):
		mkdir(save_filepath)

	# Get all .h5 files that have our first camera name
	# e.g. asdf_Cam0_1234.h5
	cam1_regex = re.compile(f'.*{camera_ids[0]}.*.h5')
	cam1_file_list = list(filter(cam1_regex.match, listdir(session_directory)))

	# Separately analyze each trial
	for cam1_file in cam1_file_list:

	    # Each video from each camera for a given trial
	    # e.g. [asdf_Cam0_1234.h5, asdf_Cam1_1234.h5, asdf_Cam2_1234.h5] 
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

	    """Thresholding with score_threshold"""

	    point_scores = [get_point_scores(join(session_directory, cam_file)) for cam_file in cam_group]
	    # stack cameras on top of each other 3 x tracks x nodes x 1
	    # (already technically this dimension... turn to nparray?)
	    # honestly just matching what was done for pose2d
	    point_scores = np.stack(point_scores, axis=0)[:, :]
	    point_scores = point_scores.reshape(n_cams, n_frames, n_nodes)

	    # Turns indices in pose2d with corresponding scores below score_threshold to nan
	    bad = point_scores < config['triangulation']['score_threshold']
	    pose2d[bad] = np.nan

	    """Get 3D data from triangulation."""
	    """Aniposelib gives us the option to triangulate with the direct linear transformation (DLT) or with RANSAC,
	    which adds an outlier rejection subroutine to the DLT. <br>
	    In addition to these 2 triangulation methods, we can further refine the 3D points via direct optimization
	    of the reprojection error.
	    def get_3d_poses(
	                    poses_2d: list,
	                    calibration_filepath: str = None,
	                    config: dict
	                    ) -> np.ndarray
	    Args:
	        poses_2d: A length # cameras list of pose matrices for a single animal. Each pose matrix is of 
	        shape (# frames, # nodes, 2, # tracks).

	        calibration_filepath: Filepath to calibration.toml.

	        config: Dictionary with user parameters for triangulation.
	        
	    Returns:
	        poses_3d: A (# frames, # nodes, 3, # tracks) that corresponds to the triangulated 3D points in the world frame. 
	    """
	    
	    #with io.capture_output() as captured:
	    
	    pose3d, errors = get_3d_poses(poses_2d=pose2d,
	                    calibration_filepath=calibration_file,
	                    config=config)

	    # Save as .mat to save_filepath
	    trial_name = cam1_file
	    save_as_mat(session_name, trial_name, marker_names, video_files, pose3d, save_filepath, errors)

def save_as_mat(session_name, trial_name, marker_names, video_files, pose3d, save_filepath, errors):
	"""
	Save pose3d data to .mat file.

	Helper function to triangulate_poses().
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