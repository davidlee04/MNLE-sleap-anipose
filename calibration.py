import importlib
import numpy as np
import re
import os.path

from collections import defaultdict

spec = importlib.util.find_spec('aniposelib')
if spec is None:
	print("aniposelib is not installed")
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup

from utils import get_files
#error info
def generate_calibration(config):
	# FIX CASE SENSITIVE Cam
	assert config is not {}, 'empty config'
	session_path = config['path']
	config_calibration = config['calibration']
	calibration_folder = session_path+'/../calibration'

	calibration_toml_path = os.path.join(calibration_folder, 'calibration.toml')

	if os.path.exists(calibration_toml_path):
		replace_prompt = input('calibration.toml already exists, do you want to replace it? Y/N')
		if replace_prompt.lower()!='y':
			print('canceling operation')
			return

	video_extension = config_calibration['video_extension']

	video_file_pattern = fr'.*\.{video_extension}$'
	
	video_files = get_files(calibration_folder, video_file_pattern)

	print(f'VIDEO FILES: {video_files}')

	cam_videos = defaultdict(list)
	cam_names = set()

	cam_regex = re.compile(config_calibration['cam_regex'])

	if 'cam_naming' in config_calibration:
		cam_regex = re.compile(config_calibration['cam_regex'], re.IGNORECASE)

	for video_file in video_files:
		name_match = cam_regex.search(video_file)
		if name_match is None:
			continue
		assert len(name_match.groups())>0, 'regex mismatch'
		name = name_match.group()
		print(f'NAME {name}')
		cam_names.add(name)
		cam_videos[name].append(video_file)

	if 'n_cams' in config_calibration:
		n_cams = config_calibration['n_cams']
		assert len(cam_names)==config_calibration['n_cams'], f'n_cams was {n_cams} but found {len(cam_names)} cameras'
	else:
		config_calibration['n_cams'] = len(cam_names)

	cam_names = sorted(cam_names)

	print(f'CAM_VIDEOS {cam_videos}')
	print(f'CAM_NAMES {cam_names}')

	
	board_videos = [sorted(cam_videos[cname]) for cname in cam_names]
	# board_vids = np.array(board_vids).reshape(3,1)
	# assert not len(board_vids)>3, 'too many calibration videos'
	# assert not len(board_vids)<3, 'too few calibration videos'
	print(f'BOARD_VIDEOS {board_videos}')
	board_size = config_calibration['board_size']
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

	cgroup = CameraGroup.from_names(cam_names, config_calibration['fisheye'])

	error, _ = cgroup.calibrate_videos(board_videos, board)

	cgroup.metadata['error'] = error.item()

	#fix this
	cgroup.dump(os.path.join(session_path, 'calibration.toml'))
