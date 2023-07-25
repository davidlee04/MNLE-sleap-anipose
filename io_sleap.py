from os.path import dirname
import re

from pathlib import PurePath

from sleap import Labels, Video


def convert(in_filepath: str, out_filepath: str, format: str = 'analysis', video: Video = None):
    """`sleap-convert` function for converting .slp to different formats.

    Args:
        in_filepath: Input slp file path
        out_filepath: Output file path for the converted file
        format: Type of conversion (must match extension of out_filepath)
        video: video

    Reads:
    * SLEAP dataset in .slp, .h5, .json, or .json.zip file
    * SLEAP "analysis" file in .h5 format
    * LEAP dataset in .mat file
    * DeepLabCut dataset in .yaml or .csv file
    * DeepPoseKit dataset in .h5 file
    * COCO keypoints dataset in .json file

    Writes:
    * SLEAP dataset (defaults to .slp if no extension specified)
    * SLEAP "analysis" file (.h5)

    You don't need to specify the input format; this will be automatically detected.

    Analysis HDF5:

    If you want to export an "analysis" h5 file, use `--format analysis`.

    The analysis HDF5 file has these datasets:

    * "track_occupancy"    (shape: tracks * frames)
    * "tracks"             (shape: frames * nodes * 2 * tracks)
    * "track_names"        (shape: tracks)
    * "node_names"         (shape: nodes)
    * "edge_names"         (shape: nodes - 1)
    * "edge_inds"          (shape: nodes - 1)
    * "point_scores"       (shape: frames * nodes * tracks)
    * "instance_scores"    (shape: frames * tracks)
    * "tracking_scores"    (shape: frames * tracks)

    Note: the datasets are stored column-major as expected by MATLAB.
    This means that if you're working with the file in Python you may want to
    first transpose the datasets so they matche the shapes described above.
    """
    assert format in ['h5', 'slp', 'json', 'analysis'], f'Incorrect conversion format {format}'

    video_callback = Labels.make_video_callback([dirname(in_filepath)])
    try:
        labels: Labels = Labels.load_file(in_filepath, video_search=video_callback)
    except TypeError:
        print("Input file isn't SLEAP dataset so attempting other importers...")
        from sleap.io.format import read

        video_path = video if video else None

        labels = read(
            in_filepath,
            for_object="labels",
            as_format="*",
            video_search=video_callback,
            video=video_path,
        )

    if format == "analysis":
        from sleap.info.write_tracking_h5 import main as write_analysis

        write_analysis(
            labels,
            output_path=out_filepath,
            labels_path=in_filepath,
            all_frames=True,
            video=video,
        )

    elif format in ("slp", "h5", "json"):
        print(f"Output SLEAP dataset: {out_filepath}")
        Labels.save_file(labels, out_filepath)

    else:
        print("You didn't specify how to convert the file.")
        print(format)
