import logging
from matplotlib import pyplot as plt
import common
import click

# Configure the global logger
logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

@click.command()
@click.option('-i', 'inputfile', type=click.Path(exists=True,resolve_path=True), required=True)
@click.option('-r', 'referencefile', type=click.Path(exists=True,resolve_path=True), required=True)
@click.option('-o', 'outputfile', type=click.Path(resolve_path=True), required=True)
@click.option('-feature', type=click.Choice(['sift','orb','akaze','brisk']), default='brisk', help='<sift|surf|orb|akaze|brisk>')
@click.option('-nfeat', type=int, default=None, help='Number of features to detect per image')
@click.option('-trs', 'target_resize_scale', type=int, default=None, help='Resize scale for target image. Changes resolution')
@click.option('-rrs', 'reference_resize_scale', type=int, default=None, help='Resize scale for reference image. Changes resolution')
@click.option('-flann/-no-flann', 'flann', default=True, help='use flann matcher')
@click.option('-gcps/-no-gcps', 'gcps', default=False, help='export gcps')
@click.option('-ratio', default=0.75, type=float, help='ratio test explained by D.Lowe')
@click.option('-plot/-no-plot', 'plot', default=False,
              help='Option to output plot with reference image, target image and matching keypoints' )
@click.option('-tileX', 'tileX', type=int, default=None, help='')
@click.option('-tileY', 'tileY', type=int, default=None, help='')
@click.option('-offsetX', 'offsetX', type=int, default=None, help='')
@click.option('-offsetY', 'offsetY', type=int, default=None, help='')
@click.option('-ki', type=click.Path(exists=True,resolve_path=True), default=None, help='keypoint inputfile')
@click.option('-kr', type=click.Path(exists=True,resolve_path=True), default=None, help='keypoint referencefile')
@click.option('-t', 'transform_type', type=click.Choice(['poly1', 'poly2', 'poly3', 'tp']),
              default='poly1', help='transform type <poly1, poly2, poly3, tps>')
@click.option('-res', 'resampleAlg',
              type=click.Choice(['nearest', 'bilinear', 'cubic', 'cubic_spline','lanczos',
                                 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3']),
              default='cubic',
              help='Resampling method <near,bilinear,cubic,cibucspline,lanczos,average,rms,mode,max,min,med,q1,q3,sum>')
@click.option('-nd', 'noData', type=int, default=None, help='No Data Value')
@click.option('-cv/-no-cv', 'convert', default=False, help='Converts input images to binary before feature detection and matching')
@click.option('-clahe/-no-clahe', 'clahe', default=False, help='Applies CLAHE histogram equilization before feature detection and matching.If both -clahe and -cv arguements are give, CLAHE equalization will be applied first.')

def automatch(inputfile, referencefile, outputfile, feature, target_resize_scale,
              reference_resize_scale, nfeat, flann, gcps, ratio, plot, tileX, tileY,
              offsetX, offsetY, ki, kr, transform_type, resampleAlg, noData, convert, clahe):
    """Simple autoreferencing tool. Calculates and executes the transform of
    target image with no georefence or very poor georeference. A reference image
    with correct georeferencing must be used. The tie points and GCP collection
    is done by combining automated feature detection algorithms. The transformation
    of the target image is done using GDAL warping tools.
    """
    logging.debug('Input file: %s', inputfile)
    logging.debug('Reference file: %s', referencefile)
    logging.debug('Output file: %s', outputfile)
    logging.debug('Feature: %s', feature)
    if target_resize_scale:
        logging.debug('Target Image Resize Scale: %d', target_resize_scale)
    if reference_resize_scale:
        logging.debug('Target Image Resize Scale: %d', reference_resize_scale)
    if flann:
        logging.debug('Matcher: FLANN')
    else:
        logging.debug('Matcher: BF')
    if gcps:
        logging.debug('GCPs export: TRUE')
    else:
        logging.debug('GCPs export: FALSE')
    logging.debug('Matching Ratio: %f', ratio)
    logging.debug('Show plot: %d', plot)
    logging.debug('Resampling method: %s', resampleAlg)
    logging.debug('Transformation method: %s', transform_type)
    
    if tileX:
        logging.debug('Horizontal Tile Size (tileX): %d', tileX)
    if tileY:
        logging.debug('Vertical Tile Size (tileY): %d', tileY)
    if offsetX:
        logging.debug('Horizontal Offset (offsetX): %d', offsetX)
    if offsetY:
        logging.debug('Vertical offset Size (offsetY): %d', offsetY)
    
    if kr:
        logging.debug('Loading reference image keypoints from: %s', kr)
    if ki:
        logging.debug('Loading input image keypoints from: %s', ki)
    
    tile = None
    offset = None
    if tileX is not None and tileY is not None:
        tile = (tileX, tileY)
        if offsetX is not None and offsetY is not None:
            offset = (offsetX, offsetY)
        else:
            offset = tile  # if no offset is set, use the tile sizes
    
    common.georef(inputfile, referencefile, outputfile,
                  feature, target_resize_scale,reference_resize_scale,
                  nfeat, flann, transform_type, resampleAlg,
                  gcps, tile, offset, ki, kr, ratio,
                  noData, plot, convert, clahe)
    logging.debug('Georeferencing Completed!. Image stored in: %s', outputfile)   
    
if __name__ == "__main__":
    automatch()
  

# =============================================================================
#     tile = None
#     offset = None
#     if tileX is not None and tileY is not None:
#         tile = (tileX, tileY)
#         if offsetX is not None and offsetY is not None:
#             offset = (offsetX, offsetY)
#         else:
#             offset = tile  # if no offset is set, use the tile sizes
#     if gcps:
#         pointsShp = outputfile + '.shp'
#         pointsTxt = outputfile + '.points'        
#     img_query, feature_name_query, kp_query, des_query = common.loadOrCompute(ki, inputfile, feature, tile,
#                                                                               offset)
#     img_train, feature_name_train, kp_train, des_train = common.loadOrCompute(kr, referencefile, feature, tile,
#                                                                               offset)
#     assert feature_name_train == feature_name_query
#     _, norm = common.init_feature(feature_name_query)
# 
#     # Match both ways
#     two_sides_matches = common.match(norm, flann, des_train, des_query, ratio)
#     projection, gcps, dst_gcps = common.getGCP(referencefile, kp_query, kp_train, two_sides_matches, min_matches=3)
#     pointsShp = None
#     pointsTxt = None
# 
#     common.saveGeoref(inputfile, outputfile, projection, transform_type, resampleAlg, gcps, dst_gcps, pointsShp, pointsTxt)
# 
# =============================================================================
