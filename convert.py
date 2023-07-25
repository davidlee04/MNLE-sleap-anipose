from os.path import split, exists, join
from os import mkdir

from utils import extract_prefix, extract_match, get_files
from io_sleap import convert
from IPython.utils import io

import re

def convert_slp_to_h5(config):
	"""Converts .slp files to .h5 files.

	Args:
		config: Dictionary representing config file
	"""

	predictions_directory = config['path']+'/predictions'
	predictions_h5_directory = config['path']+'/predictions_h5'
	print(f'Converting {predictions_directory}')
	print(f'Outputting to {predictions_h5_directory}')

	# Get all .slp files
	slp_regex_pattern = r'.*\.slp$'
	
	slp_file_paths = get_files(predictions_directory, slp_regex_pattern)
	print(f'SLEAP FILES: {slp_file_paths}')

	conversion_format = 'analysis'
	new_extension = '.analysis.h5'
	original_extension = '.predictions.slp'

	# if save location does not exist, create it
	if not exists(predictions_h5_directory):
		mkdir(predictions_h5_directory)

	# Generate .h5 files
	for slp_file_path in slp_file_paths:
		_, slp_file = split(slp_file_path)

		# camID = extract_match(in_filename, pattern = r'.*Cam\d+')

		# e.g. cam_regex = 'Cam([A-Z])'
		# cam_regex = re.compile(config['cam_regex'])
		# name_match = cam_regex.search(slp_file)

		# # no matching regex pattern
		# if name_match is None:
		# 	continue

		# assert len(name_match.groups())>0, 'no capture group in regex'
		
		# look for video extension (e.g. '.avi')
		
		if not slp_file.endswith('.predictions.slp'):
			continue

		prediction_h5_file = slp_file[0:len(slp_file)-len(original_extension)]+new_extension
		prediction_h5_file_path = join(predictions_h5_directory, prediction_h5_file)
		print(f'Saving to {prediction_h5_file_path}')
		# Convert file
		with io.capture_output() as captured: # Block unnecessary output from sleap.info.write_tracking_h5.write_occupancy_file
			convert(slp_file_path, prediction_h5_file_path, format=conversion_format)