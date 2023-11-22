from os.path import split, exists, join
from os import mkdir

from utils import get_files
from io_sleap import convert
from IPython.utils import io

def convert_slp_to_h5(path):
	"""Converts .slp files to .h5 files.

	Args:
		path: Session directory
	"""

	predictions_directory = join(path, 'predictions')
	predictions_h5_directory = join(path, 'predictions_h5')
	print(f'Converting {predictions_directory}')
	print(f'Outputting to {predictions_h5_directory}')

	# Get all .slp files
	slp_regex_pattern = r'.*\.slp$'
	slp_file_paths = get_files(predictions_directory, slp_regex_pattern)
	print(f'SLEAP FILES: {slp_file_paths}')

	conversion_format = 'analysis'
	new_extension = '.analysis.h5'
	original_extension = '.predictions.slp'

	if not exists(predictions_h5_directory):
		mkdir(predictions_h5_directory)

	# Generate .h5 files
	for slp_file_path in slp_file_paths:
		# File name separated from its path
		_, slp_file = split(slp_file_path)
		
		# Skip over files not ending in '.predictions.slp'
		if not slp_file.endswith(original_extension):
			continue

		# New file name will only have info pertaining to its trial group
		slp_file = slp_file.split('.')[0]

		# Exporting to .h5 with new name/extension
		prediction_h5_file = slp_file+new_extension
		prediction_h5_file_path = join(predictions_h5_directory, prediction_h5_file)
		print(f'Saving to {prediction_h5_file_path}')

		with io.capture_output() as captured: # Block unnecessary output from sleap.info.write_tracking_h5.write_occupancy_file
			convert(slp_file_path, prediction_h5_file_path, format=conversion_format)