# Installation
## Install miniconda
## Create conda environment
```
conda create --name <yourenvironmentname> python==3.10
```
## Activate environment
```
conda activate <yourenvironmentname>
```
## Install requirements
### Install from requirements.txt
```
pip install -r requirements.txt
```
### or install the following packages manually
```
conda install geopandas
conda install rasterio
conda install pyyaml
pip install opencv-python
conda install click
conda install -c conda-forge arosics
git clone https://github.com/leftfield-geospatial/simple-ortho.git
pip install -e simple-ortho
```

# Running
## Introduction
The installed modules are three different tools.
### Automatch
A tool for georeferencing warped satellite imagery according to a reference image. The input image can have georeference or completely lack georeference. The method used for gcp collection is traditional image matching through feature detection (SIFT, AKAZE, BRISK, ORB), reprojection and resampling is done using gdal warp functionality. Input and reference images can have different resolutions but good results can only be achieved with SIFT or AKAZE feature detection algorithms.
Execution speed depends on image pixel size, resolution an the feature detection algorithm chosen.
### Arosics
A tool used for satellite image cooregistration between input image and reference satellite imagery. Input and reference images can have different resolutions. Very robust on small shifts.
### Simple-ortho
Simple orthorectification tool for images with known DEM and camera model.

## Basic Automatch run 
Runs from conda command line:
1. get info on all arguements
```
python automatch.py --help
```
2. sample run
```
python automatch.py -i <input image directory> -r <reference image directory> -o <output image directory>
```
## Additional parameters
| Arguement full name | Arguement | type      | description                                  |
| ----------- | ------------ | --------- | -------------------------------------------- |
| --input     | `-i`           | DIRECTORY | Input directory of target image              |
| --output    | `-o`           | DIRECTORY | Output path                                  |
| --reference | `-r`           | DIRECTORY | Reference image directory                    |
| --feature   | `-feature`     | CHOICE    | Feature detection alg.[sift,orb,brisk,akaze] |
| --nfeat     | `-nfeat`       | INTEGER   | Number of features to detect per image       |
|             | `-ki`          | DIRECTORY | Available keypoints for input image          |
|             | `-kr`          | DIRECTORY | Available keypoints for reference image      |
|             | `-t`           | CHOICE    | Transform type [poly1,poly2,poly3,tp]        |
|             | `-res`         | CHOICE    | Resample type [nearest,bilinear,cubic...]    |
|             | `-nd`          | INT       | Output NoData Value                          |
|             | `-cv/-no-cv`   | BOOLEAN   | Convert images to binary format before matching |
|             | `-clahe/-no-clahe`| BOOLEAN| Apply CLAHE histogram equilization before matching |
|             |`-tileX`       | INTEGER   | Pixel X coordinates for image area subset    |
|             |`-tileY`        | INTEGER   | Pixel Y coordinates for image area subset    |
|             |`-offsetX`      | INTEGER   | X offset, in pixels, for image subset        |
|             |`-offsetY`      | INTEGER   | Y offset, in pixels, for image subset        |
| --target_resize_scale|`-trs`          | INTEGER   | Target image resize scale           |
| --reference_resize_scale|`-rrs`          | INTEGER   | Reference image resize scale     |
|             |`-flann/-no-flann`| BOOLEAN | Activation of FLANN matcher                  |
|             |`-gcps/-no-gcps`| BOOLEAN   | Export gcps for reference image              |
|             |`-ratio` | FLOAT     | Lowe's ratio value for selecting good matches|
|             |`-show/-no-show`| BOOLEAN   | Show plot with images and matched features   |
| --help      |              |           | Shows help message and exits.                |

## Basic Arosics run for local coregistration 
Runs from conda command line:
1. get info on all arguements
`python arosics_coreg_local.py --help`
2. sample run
`python arosics_coreg_local.py -i <input image directory> -r <reference image directory> -o <output image directory>`
## Additional parameters
| Arguement full name | Arguement| type      | description                                  |
| ----------- | ------------ | --------- | -------------------------------------------- |
| `--input`     | `-i`          | DIRECTORY | Input directory of target image              |
| `--output`    | `-o`          | DIRECTORY | Output path                                  |
| `--reference` | `-r`           | DIRECTORY | Reference image directory                    |
|`--gridres`|`-grid`|FLOAT|Tie point grid resolution in pixels|
|`--max_points`|`-mp`|INTEGER|Maximum number of points used to find coregistration tie poitns|
|`--window_size`|`-ws`|TUPLE OF INTEGERS|dimensions of custom matching window size|
|`--fmt_out`|`-fmt`|STRING|Raster file format for output file. Ignored if path_out is None.|
|`--projectDir`|`-dir`|STRING|Name of a project directory where to store all the output results.|
|`--r_b4match`|`-rb4`|INTEGER|band of reference image to be used for matching. Starts with 1, default 1.|
|`--s_b4match`|`-sb4`|INTEGER|band of shift image to be used for matching. Starts with 1, default 1.|
|`--max_iter`|`-mi`|INTEGER|maximum number of iterations for matching|
|`--max_shift`|`-ms`|INTEGER|maximun shift distance in reference image pixel units (default 5px)|
|`--tieP_filter_level`|`-tfl`|INTEGER|filter tie points used for shift correction in different levels (default 3)|
|`--min_reliability`|`-mr`|FLOAT|Tie point filtering minimum reliability threshold.|
|`--rs_max_outlier`|`-rsmo`|FLOAT|RANSAC tie point filtering: proportion of expected outliers|
|`--rs_tolerance`|`-rst`|FLOAT|RANSAC tie point filtering: precentage tolerance for max_outlier_precentange|
||`-ag/-no-ag`|BOOLEAN|Align input coordinate grid to the reference grid|
||`-mg/-no-mg`|BOOLEAN|Match the input pixel size to the reference pixel size|
|`--out_gsd`|`-og`|FLOAT|output pixel size in units of the reference coordinate system|
|`--target_xyGrid`|`-txy`|STRING|a list with a target x-grid and a target y-grid like [[15,45],[15,45]] as a string. This overrides 'out_gsd', 'align_grids' and 'match_gsd|
|`--resamp_alg_deshift`|`-rsd`|CHOICE|the resampling algorithm to be used for shift correction (if neccessary) valid algorithms: nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, max, min, med, q1, q3 (default: cubic)|
|`--resamp_alg_calc`|`-rsc`|CHOICE|the resampling algorithm to be used for all warping processes during calculation of spatial shifts valid algorithms: nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, max, min, med, q1, q3 (default: cubic (highly recommended))|
|`--footprint_poly_ref`|`-fpref`|STRING|footprint polygon of the reference image (WKT string or shapely.geometry.Polygon)|
|`--footprint_poly_tgt`|`-fptgt`|STRING|footprint polygon of image to be shifted (WKT string or shapely.geometry.Polygon)|
|`--data_corners_ref`|`-dcref`|STRING|map coordinates of data corners within reference image as string [float,float,float,float]. ignored if footprint_poly_ref is given.|
|`--data_corners_tgt`|`-dctgt`|STRING|map coordinates of data corners within image to be shifted as string [float,float,float,float]. ignored if footprint_poly_tgt is given.|
|`--outFillVal`|`-fv`|INTEGER|if given the generated tie point grid is filled with this value in case no match could be found during co-registration (default: -9999)|
|`--nodata`|`-nd`|TUPLE OF INTEGERS|no data values for reference image and image to be shifted. input in the form of a tuple (int,int)|
||`-cc/-no-cc`|BOOLEAN|calculate true positions of the dataset corners in order to get a useful matching window position within the actual image overlap (default: True; deactivated if 'data_corners_im0' and 'data_corners_im1' are given)|
||`-bws/-no-bws`|BOOLEAN|use binary X/Y dimensions for the matching window (default: True)|
||`-fqw/-no-fqw`|BOOLEAN|use binary X/Y dimensions for the matching window (default: True)|
||`-cpus`|INTGEGER|umber of CPUs to use during calculation of tie point grid (default: None, which means 'all CPUs available')|
||`-prog/-no-prob|BOOLEAN|how progress bars (default: True)|
||`-v/-no-v`|BOOLEAN|verbose mode (default: False)|
||`-q/-no-q`|BOOLEAN|quiet mode (default: False)|
||`-ie/-no-ie`|BOOLEAN|Ignore Errors. Useful for batch processing. (default: False)|
||`-show/-no-show`|BOOLEAN|plot target image with tie points|
| `--help`     |              |           | Shows help message and exits.                |

## Basic simple_ortho run
`simple-ortho [-h] [-od <ortho_dir>] [-rc <config_path>] [-wc <config_path>] [-v {1,2,3,4}] src_im_file [src_im_file ...] dem_file pos_ori_file`

## Required Arguements
Argument  | Description
----------|--------------
`-src_im_file` | One or more path(s) and or wildcard(s) specifying the source unrectified image file(s).
`-dem_file` | Path to a DEM, that covers all image(s) specified by `src_im_file`.  
`-pos_ori_file` | Path to a text file specifying the camera position and orientation for  all image(s) specified by `src_im_file`.  See [camera position and orientation section](#camera-position-and-orientation) for more detail. 

##Optional arguments
Argument | Arguement Full Name| Description
---------|-----------|------------
`-h` | `--help` | Print help and exit.
`-od` `<ortho_dir>` | `--ortho-dir` `<ortho_dir>` | Write orthorectified images to `<ortho_dir>` (default: write to source directory).
`-rc` `<config_path>` | `--read_conf` `<config_path>` | Read a custom configuration from the specified `<config_path>`.  If not specified, sensible defaults are read from [config.yaml](config.yaml).  See [configuration](#configuration) for more details.  
`-wc` `<config_path>` | `--write_conf` `<config_path>` | Write current configuration to  `<config_path>` and exit.
`-v` `{1,2,3,4}` | `--verbosity {1,2,3,4}` | Set the logging level (lower means more logging).  1=debug, 2=info, 3=warning, 4=error (default: 2).