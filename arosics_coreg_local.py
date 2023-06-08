import os
from arosics import COREG_LOCAL
import click
import logging
import re


# Configure the global logger

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

@click.command()
@click.option('-i', 'inputfile', type=click.Path(exists=True), required=True, help='source path of image to be shifted (any GDAL compatible image format is supported)')
@click.option('-r', 'referencefile', type=click.Path(exists=True), required=True, help='source path of reference image (any GDAL compatible image format is supported)')
@click.option('-o', 'path_out', type=click.Path(), required=True, default=None, help="maximum number of points used to find coregistration tie points")
@click.option('-grid', 'grid_res', type=float, help="tie point grid resolution in pixels of the target image")
@click.option('-mp', 'max_points',nargs=2,type=int, default=None, help='maximum number of points used to find coregistration tie points')
@click.option('-ws', 'window_size',nargs=2, type=click.Tuple([int, int]), default=(256,256), help='dimensions of custom matching window size [pixels]. Input as multiple arguements: int int (default: 256 256)')
@click.option('-fmt', 'fmt_out', type=str, default='ENVI', help="raster file format for output file. ignored if path_out is None. Can be any GDAL compatible raster file format (e.g. 'ENVI', 'GTIFF'; default: ENVI). Refer to https://gdal.org/drivers/raster/index.html to get a full list of supported formats")
#@click.option('-outopt', 'out_crea_options', multiple=True, default=None, help='GDAL creation options for the output image, e.g. ["QUALITY=80", "REVERSIBLE=YES", "WRITE_METADATA=YES"]')
@click.option('-dir', 'projectDir', type=str, default=None, help="name of a project directory where to store all the output results. If given, name is inserted into all automatically generated output paths")
@click.option('-rb4', 'r_b4match', type=int, default=1, help="band of reference image to be used for matching (starts with 1; default: 1)")
@click.option('-sb4', 's_b4match', type=int, default=1, help="band of shift image to be used for matching (starts with 1; default: 1)")
@click.option('-mi', 'max_iter', type=int, default=5, help="maximum number of iterations for matching (default: 5)")
@click.option('-ms', 'max_shift', type=int, default=5, help="maximum shift distance in reference image pixel units (default: 5 px)")
@click.option('-tfl', 'tieP_filter_level', type=int, default=3, help="filter tie points used for shift correction in different levels (default: 3).NOTE: lower levels are also included if a higher level is chosen - Level 0: no tie point filtering. - Level 1: Reliablity filtering, filter all tie points out that have a low reliability according to internal tests. - Level 2: SSIM filtering, filters all tie points out where shift correction does not increase image similarity within matching window (measured by mean structural similarity index). - Level 3: RANSAC outlier detection")
@click.option('-mr', 'min_reliability', type=float, default=60, help="Tie point filtering: minimum reliability threshold, below which tie points are marked as false-positives (default: 60%) - accepts values between 0% (no reliability) and 100 % (perfect reliability) HINT: decrease this value in case of poor signal-to-noise ratio of your input data")
@click.option('-rsmo', 'rs_max_outlier', type=float, default=10, help="RANSAC tie point filtering: proportion of expected outliers (default: 10%).")
@click.option('-rst', 'rs_tolerance', type=float, default=2.5, help="RANSAC tie point filtering: percentage tolerance for max_outlier_percentage (default: 2.5%).")
@click.option('-ag/-no-ag', 'align_grids', default=True, help="True: align the input coordinate grid to the reference (does not affect the output pixel size as long as input and output pixel sizes are compatible (5:30 or 10:30 but not 4:30), default = True")
@click.option('-mg/-no-mg', 'match_gsd', default=False, help="True: match the input pixel size to the reference pixel size, default = False")
@click.option('-og', 'out_gsd', type=float, default=None, help="output pixel size in units of the reference coordinate system (default = pixel size of the input array), given values are overridden by match_gsd=True")
@click.option('-txy', 'target_xyGrid', type=str, default=None, help="a list with a target x-grid and a target y-grid like [[15,45],[15,45]] as a string. This overrides 'out_gsd', 'align_grids' and 'match_gsd'")
@click.option('-rsd', 'resamp_alg_deshift',
              type=click.Choice(['nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos','average', 'mode', 'max', 'min', 'med', 'q1', 'q3'], case_sensitive=False),
              default='cubic',
              help="the resampling algorithm to be used for shift correction (if neccessary) valid algorithms: nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, max, min, med, q1, q3 (default: cubic)")
@click.option('-rsc', 'resamp_alg_calc', type=click.Choice(['nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos','average', 'mode', 'max', 'min', 'med', 'q1', 'q3'], case_sensitive=False),
              default='cubic',
              help="the resampling algorithm to be used for all warping processes during calculation of spatial shifts valid algorithms: nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, max, min, med, q1, q3 (default: cubic (highly recommended))")
@click.option('-fpref', 'footprint_poly_ref', type=str, default=None, help="footprint polygon of the reference image (WKT string or shapely.geometry.Polygon), e.g. 'POLYGON ((299999 6000000, 299999 5890200, 409799 5890200, 409799 6000000, 299999 6000000))")
@click.option('-fptgt', 'footprint_poly_tgt', type=str, default=None, help="footprint polygon of image to be shifted (WKT string or shapely.geometry.Polygon), e.g. 'POLYGON ((299999 6000000, 299999 5890200, 409799 5890200, 409799 6000000, 299999 6000000))")
@click.option('-dcref', 'data_corners_ref', type=str, default=None, help="map coordinates of data corners within reference image as string [float,float,float,float]. ignored if footprint_poly_ref is given.")
@click.option('-dctgt', 'data_corners_tgt', type=str, default=None, help="map coordinates of data corners within image to be shifted as string [float,float,float,float]. ignored if footprint_poly_tgt is given.")
@click.option('-fv', 'outFillVal', type=int, default=-9999, help="if given the generated tie point grid is filled with this value in case no match could be found during co-registration (default: -9999)")
@click.option('-nd', 'nodata',nargs=2, type=click.Tuple([int, int]), default=(None,None), help="no data values for reference image and image to be shifted. input as multiple arguements: int int")
@click.option('-cc/-no-cc', 'calc_corners', default=True, help="calculate true positions of the dataset corners in order to get a useful matching window position within the actual image overlap (default: True; deactivated if 'data_corners_im0' and 'data_corners_im1' are given)")
@click.option('-bws/-no-bws', 'binary_ws', default=True, help="use binary X/Y dimensions for the matching window (default: True)")
@click.option('-fqw/-no-fqw', 'force_quadratic_win', default=True, help="use binary X/Y dimensions for the matching window (default: True)")
@click.option('-cpus', 'CPUs', type=int, default=None, help="number of CPUs to use during calculation of tie point grid (default: None, which means 'all CPUs available')")
@click.option('-prog/-no-prob', 'progress', default=True, help="how progress bars (default: True)")
@click.option('-v/-no-v', 'v', default=False, help="verbose mode (default: False)")
@click.option('-q/-no-q', 'q', default=False, help="quiet mode (default: False)")
@click.option('-ie/-no-ie', 'ignore_errors', default=False, help="Ignore Errors. Useful for batch processing. (default: False)")
@click.option('-plot/-no-plot', 'plot', default=False, help="plot target image with tie points")
@click.option('-gcps/-no-gcps', 'gcps', default=False, help='export gcps')

# =============================================================================
# im_target = r'C:\Users\msi2\Desktop\SPACE4\Sample_images\arosics_test\autoref_test_diff_T35TKF_20210212T092029_B08.tif'
# im_reference  = r'C:\Users\msi2\Desktop\SPACE4\Sample_images\arosics_test\T35TKF_20210212T092029_B04.jp2'
# outputfile = r'C:\Users\msi2\Desktop\SPACE4\Sample_images\arosics_test\coregistered_T35TKF_20210212_B08.tif'
# =============================================================================

def arosics_coreg_local(referencefile,inputfile,grid_res,max_points,window_size,
                        path_out,fmt_out,projectDir,r_b4match,s_b4match,
                        max_iter,max_shift,tieP_filter_level,min_reliability,rs_max_outlier,
                        rs_tolerance,align_grids,match_gsd,out_gsd,target_xyGrid,
                        resamp_alg_deshift,resamp_alg_calc,footprint_poly_ref,footprint_poly_tgt,
                        data_corners_ref,data_corners_tgt,outFillVal,nodata,calc_corners,
                        binary_ws,force_quadratic_win,CPUs,progress,v,q,ignore_errors,plot,gcps):
    
    logging.debug('Input file: %s', inputfile)
    logging.debug('Reference file: %s', referencefile)    

    if target_xyGrid is not None:
        target_xyGrid = re.findall(r'\d+',target_xyGrid)
        target_xyGrid = [[int(target_xyGrid[0]),int(target_xyGrid[1])],
                         [int(target_xyGrid[2]),int(target_xyGrid[3])]]
    
    if data_corners_ref is not None:
        data_corners_ref = re.findall(r'[+-]?[0-9]+[.]?[0-9]+',data_corners_ref)
        for c,i in enumerate(data_corners_ref):
            data_corners_ref[c]=float(i)
    
    if data_corners_tgt is not None:
        data_corners_tgt = re.findall(r'[+-]?[0-9]+[.]?[0-9]+',data_corners_tgt)
        for c,i in enumerate(data_corners_ref):
            data_corners_ref[c]=float(i)
    
    kwargs = {'grid_res': grid_res,
              'max_points': max_points,
              'window_size': window_size,
              'path_out': path_out,
              'fmt_out': fmt_out,
              #'out_crea_options': out_crea_options,
              'projectDir': projectDir,
              'r_b4match': r_b4match,
              's_b4match': s_b4match,
              'max_iter': max_iter,
              'max_shift': max_shift,
              'tieP_filter_level': tieP_filter_level,
              'min_reliability': min_reliability,
              'rs_max_outlier': rs_max_outlier,
              'rs_tolerance': rs_tolerance,
              'align_grids': align_grids,
              'match_gsd': match_gsd,
              'out_gsd': out_gsd,
              'target_xyGrid': target_xyGrid,
              'resamp_alg_deshift': resamp_alg_deshift,
              'resamp_alg_calc': resamp_alg_calc,
              'footprint_poly_ref': footprint_poly_ref,
              'footprint_poly_tgt': footprint_poly_tgt,
              'data_corners_ref': data_corners_ref,
              'data_corners_tgt': data_corners_tgt,
              'outFillVal': outFillVal,
              'nodata': nodata,
              'calc_corners': calc_corners,
              'binary_ws': binary_ws,
              'force_quadratic_win': force_quadratic_win,
              'CPUs': CPUs,
              'progress': progress,
              'v': v,
              'q': q,
              'ignore_errors': ignore_errors
    }
    
    logging.debug('Arguments: %s', kwargs)

    # Calculate Shifts
    CRL = COREG_LOCAL(im_ref=referencefile, im_tgt=inputfile ,**kwargs)
    CRL.correct_shifts()
    
    if plot:
        # Visualize Tie Points
        CRL.view_CoRegPoints(figsize=(20,20), backgroundIm='tgt',
                             showFig=False, savefigPath=path_out[0:-4]+'.png')
    if gcps:
        # Visualize Tie Points
        CRL.tiepoint_grid.to_PointShapefile(path_out=path_out[0:-4]+'.shp')
if __name__ == "__main__":

    arosics_coreg_local()
    