# MNLE-sleap-anipose

A Jupyter Notebook for the SLEAP-Anipose pipeline.  

---

## Configuration file parameters

Every session requires a `config.toml` file 

### General Parameters  
- **path:** a string specifying the path of the session directory (see folder structure) [*required*]

### Calibration Parameters
- **cam_naming:** Easy specification of camera video names. Choose between `"letter"` or `"number"`.

    `"letter"` will search file names with "Cam" (case-insensitive) followed by a single letter. e.g. CamA, Camb, camc, etc...
  
    `"number"` will search file names with "Cam" (case-insensitive) followed by a string of digits. e.g. Cam0, Cam00, cam1, cAm100, etc...

---

- **cam_regex:** Regex specification of camera video names. If you have a more rigid or specific naming convention, this would be used.

    Note: cam_naming's `"letter"` uses `"Cam[A-Za-z]"` and `"number"` uses `"Cam\d+"`.

> config.toml requires at least one of the above two but not both. cam_naming will override cam_regex if both are present.
- **board_type:** type of board used for calibration (`"checkerboard"` or `"charuco"`) [*default `"charuco"`*]

- **board_size:** Width and height of the grid as a length 2 array. e.g. `[5, 5]`
- **board_square_side_length:** Square side length. `int` or `float` e.g. `5`
- **fisheye:** Specifies if videos used fisheye lens. `boolean`

*The following parameters are only necessary for ChArUco boards.*

- **board_marker_length**: Length of marker side. `int` or `float` e.g. `3.75`
- **board_marker_bits:** Number of bits in the markers. `int` e.g. `4`
- **board_marker_dict_number:** Number of markers in the dictionary. `int` e.g. `100`
> `board_marker_bits` and `board_marker_dict_number` is based on aniposelib's implementation of the ChArUco board. See <https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/boards.py#L545>

### 2D Filter Parameters
- **type:** Type of filter to apply. Choose between `"median"` or `"viterbi"`.
> Currently only the `"median"` filter is properly supported.
- **medfilt:** Length of median filter. `int` e.g. `15`
- **score_threshold:** Prediction score threshold below which labels are removed. `float` e.g. `0.5`
- **offset_threshold:** Offset from median filter to classify as a jump. `int` e.g. `10`
- **spline:** Specifies whether to use cubic spline interpolation instead of linear. `boolean`

### Triangulation Parameters
- **ransac:** Specifies whether to use ransac optimization. Otherwise, will use DLT. `boolean`
- **optim:** Specifies whether to apply 3D filters. `boolean`
- **constraints:** Pairs of joints to impose smoothness and spatial constraints between. 2D `array` containing pairs of joints. If omitted, is empty.
- **
