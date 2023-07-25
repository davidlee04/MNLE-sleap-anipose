# MNLE-sleap-anipose

A Jupyter Notebook for the SLEAP-Anipose pipeline.  

---

## Configuration file parameters

Every session requires a `config.toml` file 

### General Parameters  
- **path:** A string specifying the path of the session directory (see folder structure)

### Calibration Parameters
- **cam_naming:** Easy specification of camera video names. Choose between `"letter"` or `"number"`.

    `"letter"` will search file names with "Cam" (case-insensitive) followed by a single letter. e.g. CamA, Camb, camc, etc...
  
    `"number"` will search file names with "Cam" (case-insensitive) followed by a string of digits. e.g. Cam0, Cam00, cam1, cAm100, etc...

- **cam_regex:** Regex specification of camera video names. If you have a more rigid or specific naming convention, this would be used.

    Note: cam_naming's `"letter"` uses `"Cam[A-Za-z]"` and `"number"` uses `"Cam\d+"`.

> config.toml requires at least one of the above two but not both. cam_naming will override cam_regex if both are present.
- **board_type:** Type of board used for calibration. Choose between `"checkerboard"` or `"charuco"`.
- **board_size:** Width and height of the grid as a length 2 array. e.g. `[5, 5]`
- **board_square_side_length:** Square side length. `int` or `float` e.g. `5`

*The following parameters are only necessary for ChArUco boards.*

- **board_marker_length**: Length of marker side. `int` or `float` e.g. `3.75`
- **board_marker_bits:** Number of bits in the markers. `int` e.g. `4`
- **board_marker_dict_number:** Number of markers in the dictionary. `int` e.g. `100`
> Valid `board_marker_bits` and `board_marker_dict_number` is based on aniposelib's implementation of the ChArUco board. See <https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/boards.py#L545>
