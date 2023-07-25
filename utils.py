from os.path import exists, isdir, isfile, split, join
from os import listdir
import re


def extract_prefix(in_string: str, delim_pattern: str = r'\.mp4') -> str:
    """Find delim_pattern in in_string and return prefix before delim.
    If not match is found, return entire in_string."""
    assert len(in_string) > 0, "Input string is empty"
    assert len(delim_pattern) > 0, "Pattern string is empty"
    # search for pattern in string
    re_delim = re.search(delim_pattern, in_string)
    if re_delim is None:
        # pattern not found
        out_str = in_string
    else:
        # extract prefix before pattern
        out_str = in_string[:re_delim.span()[0]]
    return out_str


def extract_match(in_string: str, pattern: str = r'.*_cam\d+') -> str:
    """Find pattern in in_string and return match.
    If not match is found, return None."""
    assert len(in_string) > 0, "Input string is empty"
    assert len(pattern) > 0, "Pattern string is empty"
    # search for pattern in string
    re_match = re.search(pattern, in_string)
    if re_match is None:
        # pattern not found
        out_str = None
    else:
        # extract prefix before pattern
        out_str = in_string[re_match.span()[0]:re_match.span()[1]]
    return out_str


def get_files(path: str, file_pattern: str = r'.*\.slp') -> list:
    """Find all matching files in path that match file_pattern and return list of complete file paths."""
    assert len(path) > 0, "Path string is empty"
    assert exists(path), "Path provided does not exist"
    assert isdir(path), "Path provided is not a valid file directory"
    assert len(file_pattern) > 0, "Pattern string is empty"
    base_file_list = listdir(path)
    out_file_list = []
    for file in base_file_list:
        match = re.findall(file_pattern, file)
        if len(match) > 0:
            out_file_list.append(join(path, file))
    return out_file_list
