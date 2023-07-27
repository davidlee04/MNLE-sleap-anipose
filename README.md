# MNLE-sleap-anipose

A Jupyter Notebook for the SLEAP-Anipose pipeline.  

---

## Configuration file parameters

Every session requires a `config.toml` file

Any parameters with a default value can be omitted.

### General Parameters  
- **path:** Path to session directory (see folder structure).

    `string`

    *required*

### Calibration Parameters
- **cam_naming:** Easy specification of camera video names.

    Choose between `"letter"` or `"number"`

    - `"letter"` will search file names with "Cam" (case-insensitive) followed by a single letter. e.g. CamA, Camb, camc, etc...
  
    - `"number"` will search file names with "Cam" (case-insensitive) followed by a string of digits. e.g. Cam0, Cam00, cam1, cAm100, etc...

- **cam_regex:** Regex specification of camera video names. If you have a more specific naming convention or want case-sensitivity, this would be used.

    `string`

    Note: cam_naming's `"letter"` uses `"Cam[A-Za-z]"` and `"number"` uses `"Cam\d+"`.

> config.toml requires at least one of the above two but not necessarily both. cam_naming will override cam_regex if both are present.

- **n_cams:** Number of cameras used. If present will compare how many unique cameras the program detects from the video names to this number    and throw an error if there's a mismatch. e.g. CamA, CamB, CamC from your videos, but you put `n_cams` as 4. Good for making sure things are working as they should.

    `int`

    *optional*

- **video_extension:** Video extension of your calibration videos.

    `string`

    *default:* `"mp4"`
  
- **board_type:** Type of board used for calibration.

    `"checkerboard"` or `"charuco"`

    *default:* `"charuco"`

- **board_size:** Width and height of the grid.

    Length 2 array

    *default:* `[5, 5]`
  
- **board_square_side_length:** Square side length.

    `int` or `float`
  
    *default:* `5`
  
  
- **fisheye:** Specifies if videos used fisheye lens.

    `boolean`

    *default:* `false`

*The following parameters are only relevant to ChArUco boards.*

- **board_marker_length**: Length of marker side.

    `int` or `float`

    *default:* `3.75`

- **board_marker_bits:** Number of bits in the markers.

    `int`

    *default:* `4`

- **board_marker_dict_number:** Number of markers in the dictionary.

    `int`

    *default:* `100`

> Valid `board_marker_bits` and `board_marker_dict_number` combinations are based on aniposelib's implementation of the ChArUco board. See <https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/boards.py#L545>.

### 2D Filter Parameters
- **type:** Type of filter to apply.
  
    Choose between `"median"` or `"viterbi"`  
    > Currently only the `"median"` filter is properly supported.

    `string`

    *default* `"median"`

- **medfilt:** Length of median filter.

    `int`

    *default:* `15`

- **score_threshold:** Prediction score threshold below which labels are removed.

    `float`

    *default:* `0.5`

- **offset_threshold:** Offset threshold from median filter to remove label.

    `int`

    *default:* `10`

- **spline:** Specifies whether to use cubic spline interpolation instead of linear.

    `boolean`

    *default:* `false`

### Triangulation Parameters
- **ransac:** Specifies whether to use ransac optimization. Otherwise, will use DLT.

    `boolean`

    *default:* `false`

- **optim:** Enables optimization and applies 3D filters.

    `boolean`

    *default:* `false`

- **score_threshold:** Prediction score threshold below which labels are removed.

    `float`

    *default:* `0.6`

- **progress:** Show progress bar.

    `boolean`

    *default:* `true`

*The following parameters are only relevant if `optim` is true.*

- **constraints:** Pairs of joints to impose smoothness and spatial constraints between.
  
    2D `array` of strings (pairs of joints)

    *default:* `[]`

- **scale_smooth:** Strength of enforcement of the smoothing constraints.

    `int`

    *default:* `4`

- **scale_length:** Strength of enforcement of the spatial constraints.

    `int`

    *default:* `2`

- **scale_length_weak:**

    `float`

    *default:* `0.5`

- **reproj_error_threshold:**

    `int`

    *default:* `15`

- **reproj_loss:**

    `string`

    *default:* `soft_l1`

- **n_deriv_smooth:**

    `int`

    *default:* `1`
