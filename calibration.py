import importlib
import numpy as np
import re

from os.path import join, exists

from utils import get_files

from collections import defaultdict

spec = importlib.util.find_spec('aniposelib')
if spec is None:
	print("aniposelib is not installed")
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import CameraGroup

def generate_calibration(config):
	"""
	Generates calibration.toml file from videos in calibration folder.

	Args:
		config: dictionary representing config.toml
	"""

	session_path = config['path']
	config_calibration = config['calibration']

	calibration_folder = session_path+'/../calibration'
	calibration_toml_path = join(calibration_folder, 'calibration.toml')

	# calibration.toml already exists! prompt user for replacement
	if exists(calibration_toml_path):
		replace_prompt = input('calibration.toml already exists, do you want to replace it? Y/N')
		if replace_prompt.lower()!='y':
			print('canceling operation')
			return

	video_extension = config_calibration['video_extension']
	video_file_pattern = fr'.*\.{video_extension}$'

	# Array of video_files matching our video_extension
	video_files = get_files(calibration_folder, video_file_pattern)
	print(f'VIDEO FILES: {video_files}')

	# Code from anipose repo

	# Dictionary with keys being cam_name and values being array of corresponding videos
	cam_videos = defaultdict(list)
	cam_names = set()

	cam_regex = re.compile(config_calibration['cam_regex'])

	# If user uses cam_naming, be case-insensitive
	if 'cam_naming' in config_calibration:
		cam_regex = re.compile(config_calibration['cam_regex'], re.IGNORECASE)

	# Extracting cam_names and building cam_videos
	for video_file in video_files:
		name_match = cam_regex.search(video_file)
		if name_match is None:
			continue

		name = name_match.group()
		print(f'NAME {name}')
		cam_names.add(name)
		cam_videos[name].append(video_file)

	# If user puts in n_cams, we do a sanity check to make sure the number of camera_names extracted matches n_cams
	if 'n_cams' in config_calibration:
		n_cams = config_calibration['n_cams']
		assert len(cam_names)==config_calibration['n_cams'], f'n_cams was {n_cams} but found {len(cam_names)} cameras'
	else:
		# Otherwise, set it ourselves. Might use later?
		config_calibration['n_cams'] = len(cam_names)

	# Maybe replace with cam_align?
	cam_names = sorted(cam_names)

	print(f'CAM_VIDEOS {cam_videos}')
	print(f'CAM_NAMES {cam_names}')
	
	# board_videos is a 2d array, each element being a list of videos associated with that camera_name
	board_videos = [sorted(cam_videos[cname]) for cname in cam_names]
	print(f'BOARD_VIDEOS {board_videos}')

	board_size = config_calibration['board_size']
	# board_size must be a length 2 array (width x height)
	assert len(board_size) == 2, 'invalid board_size'
	board_dimensions = (board_size[0], board_size[1])

	if config_calibration['board_type']=='charuco':
		kwargs = {
			'marker_bits': config_calibration['board_marker_bits'],
			'square_length': config_calibration['board_square_side_length'], 
			'marker_length': config_calibration['board_marker_length'],
			'dict_size': config_calibration['board_marker_dict_number']
		}
		board = CharucoBoard(*board_dimensions, **kwargs)
	elif config_calibration['board_type']=='checkerboard':
		board = Checkerboard(*board_dimensions, square_length=config_calibration['board_square_side_length'])
	else:
		print('invalid board_type')
		return

	# Outputting to calibration.toml
	cgroup = CameraGroup.from_names(cam_names, config_calibration['fisheye'])

	error, _ = cgroup.calibrate_videos(board_videos, board)
	cgroup.metadata['error'] = error.item()

	# change output directory to calibration folder
	cgroup.dump(join(calibration_folder, 'calibration.toml'))
