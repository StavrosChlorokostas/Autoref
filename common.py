import click
import logging
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import math
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import logging
import datetime
import pickle
from os import environ


FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH = 6

def init_feature(name,nfeat):
    if name == 'sift':
        if nfeat is not None:
            detector = cv.SIFT_create(nfeat)
            norm = cv.NORM_L2
        else:
            detector = cv.SIFT_create()
            norm = cv.NORM_L2
    # SURF patent not yet expired
    #elif name == 'surf':
    #    detector = cv.xfeatures2d.SURF_create(800)
    #    norm = cv.NORM_L2
    elif name == 'orb':
        if nfeat is not None:
            detector = cv.ORB_create(nfeat)
            norm = cv.NORM_HAMMING
        else:
            detector = cv.ORB_create()
            norm = cv.NORM_HAMMING
    elif name == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif name == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    return detector, norm


def init_matcher(flann, norm, checks=100):
    search_params = {'checks': checks}  # or pass empty dictionary
    if flann:
        logging.debug("%s : Initializing FLANN Matcher", datetime.datetime.now())
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
        matcher = cv.FlannBasedMatcher(flann_params, search_params)  # bug : need to pass empty dict (#1329)
    else:
        logging.debug("%s : Initializing BF Matcher", datetime.datetime.now())
        matcher = cv.BFMatcher(norm, crossCheck=False)
    return matcher

def getMatches(matcher, des1, des2, ratio):
    # FLANN parameters
    #    FLANN_INDEX_KDTREE = 0
    #    index_params = {'algorithm': FLANN_INDEX_KDTREE,
    #                    'trees': 5}
    #    search_params = {'checks': 100}   # or pass empty dictionary
    #
    #    logging.info("FlannBasedMatcher start")
    #    flann = cv.FlannBasedMatcher(index_params, search_params)
    logging.info("%s : Matching Features with FlannBasedMatcher Knn Match", datetime.datetime.now())
    # matches = matcher.knnMatch(np.asarray(des1,np.float32), np.asarray(des2,np.float32), k=2)
    # matches = matcher.match(np.asarray(des1), np.asarray(des2))
    # sorted_matches = sorted(matches,key=lambda t: t.distance)
    matches = matcher.knnMatch(queryDescriptors=np.asarray(des1), trainDescriptors=np.asarray(des2), k=2)
    # Apply ratio test
    matches = [m for m in matches if len(m) == 2] #Keep only match pairs
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches

def two_sides_match(norm, flann, des_train, des_query, ratio):
    # Match both ways
    matcher = init_matcher(flann, norm)
    if ratio is None:
        ratio = 0.75
    matches_train = getMatches(matcher, des_train, des_query, ratio)
    matches_query = getMatches(matcher, des_query, des_train, ratio)
    # Only keep matches that are present in both ways
    two_sides_matches = [m for m in matches_train if
                         any(mm.queryIdx == m.trainIdx and mm.trainIdx == m.queryIdx for mm in matches_query)]
    logging.debug("%s : Found %d Matches", datetime.datetime.now(), len(two_sides_matches))
    return two_sides_matches

def one_side_match(norm, flann, des_train, des_query, ratio):
    # Match both ways
    matcher = init_matcher(flann, norm)
    if ratio is None:
        ratio = 0.75
    matches = getMatches(matcher, des_train, des_query, ratio)
    # Only keep matches that are present in both ways
    return matches

def getKeyPointsAndDescriptors(detector, img, tile, offset, n_features=0):
    if tile and offset:
        img_shape = img.shape
        nbTilesY = int(math.ceil(img_shape[0] / (offset[1] * 1.0)))
        nbTilesX = int(math.ceil(img_shape[1] / (offset[0] * 1.0)))
        array = []
        point_set = set()
        # only used to have a fast 'contains' test before adding keypoints in order to avoid duplicates
        for i in range(nbTilesY):
            logging.debug("%s : %d/%d", datetime.datetime.now(), i, nbTilesY)
            for j in range(nbTilesX):
                x_min = offset[0] * j
                y_min = offset[1] * i
                cropped_img = img[
                              y_min:min(offset[1] * i + tile[1], img_shape[0]),
                              x_min:min(offset[0] * j + tile[0], img_shape[1])]
                kp, des = detector.detectAndCompute(cropped_img, None)
                logging.debug("%s : Found %d points", datetime.datetime.now(), len(kp))
                if len(kp) > 0:
                    for z in zip(kp, des):
                        pt = z[0].pt
                        z[0].pt = (pt[0] + x_min, pt[1] + y_min)
                        if z[0].pt not in point_set:
                            point_set.add(z[0].pt)
                            array.append(z)
                        # array.append(z)
        sorted_array = sorted(array, key=lambda t: t[0].response, reverse=True)
        if n_features > 0:
            sorted_array = (array[0][:n_features],array[1][:n_features])
        return [e[0] for e in sorted_array], [e[1] for e in sorted_array]
    else:
        kp,des = detector.detectAndCompute(img, None)
        logging.debug("%s : Found %d points", datetime.datetime.now(), len(kp))
        array = [i for i in zip(kp,des)]
        sorted_array = sorted(array, key=lambda t: t[0].response, reverse=True)
        if n_features > 0:
            sorted_array = array[:n_features]
        return [e[0] for e in sorted_array], [e[1] for e in sorted_array]

def getImageKeyPointsAndDescriptors(image_file, detector, tile, offset,resize_scale=None,
                                    convert_to_binary=False, apply_clahe=False, n_features=10000):
    input_img = cv.imread(image_file, cv.IMREAD_ANYDEPTH)  # queryImage (IMREAD_COLOR flag=cv.IMREAD_GRAYSCALE to force grayscale)
    if  resize_scale is not None:
        width = int(input_img.shape[1] * resize_scale / 100)
        height = int(input_img.shape[0] * resize_scale / 100)
        dim = (width, height)
        input_img = cv.resize(input_img, dim, interpolation=cv.INTER_LANCZOS4)
        logging.debug('%s: Image resized to %d x %d pixels', datetime.datetime.now(),width, height)
    # img = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
    img = cv.normalize(input_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    norm_img = cv.normalize(input_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    if apply_clahe or convert_to_binary:
        blur = cv.GaussianBlur(img, (5, 5), 0)
    if apply_clahe:
        img = apply_CLAHE(blur)
    if convert_to_binary:
        if apply_clahe:
            img = getBinImage(img)
        else:
            img = getBinImage(blur)
    
    kp, des = getKeyPointsAndDescriptors(detector, img, tile, offset, n_features)
    logging.info("%s : Feature detection completed", datetime.datetime.now())
    return norm_img, img, kp, des


def loadOrCompute(ki, input_file, feature_name, resize_scale, nfeat, tile, offset, convert, clahe):
    if ki is None:
        if feature_name is None:
            feature_name = 'brisk'
        detector, norm = init_feature(feature_name,nfeat)
        input_img, img, kp, des = getImageKeyPointsAndDescriptors(input_file, detector, tile, offset, resize_scale, convert, clahe)
        return input_img, img, feature_name, kp, des
    else:
        return loadKeyPoints(ki)

def getBinImage(image):
    # Otsu's thresholding after a gaussian blur
    th, img = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return img

def apply_CLAHE(image):
    # Clahe histogram equilization after gaussian blur
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(image)
    return img


def saveGCPsAsShapefile(gcps, srs, output_file):
    out_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(output_file)
    out_lyr = out_ds.CreateLayer('gcps', geom_type=ogr.wkbPoint, srs=srs)
    out_lyr.CreateField(ogr.FieldDefn('Id', ogr.OFTString))
    out_lyr.CreateField(ogr.FieldDefn('Info', ogr.OFTString))
    out_lyr.CreateField(ogr.FieldDefn('X', ogr.OFTReal))
    out_lyr.CreateField(ogr.FieldDefn('Y', ogr.OFTReal))
    for i in range(len(gcps)):
        f = ogr.Feature(out_lyr.GetLayerDefn())
        f.SetField('Id', gcps[i].Id)
        f.SetField('Info', gcps[i].Info)
        f.SetField('X', gcps[i].GCPPixel)
        f.SetField('Y', gcps[i].GCPLine)
        f.SetGeometry(ogr.CreateGeometryFromWkt('POINT(%f %f)' % (gcps[i].GCPX, gcps[i].GCPY)))
        out_lyr.CreateFeature(f)


def saveGCPsAsText(gcps, output_file):
    file = open(output_file, "w+")
    file.write("mapX,mapY,pixelX,pixelY,enable,dX,dY,residual\n")
    for i in range(len(gcps)):
        file.write("%f,%f,%f,%f,1,0,0,0\n" %(gcps[i].GCPX, gcps[i].GCPY, gcps[i].GCPPixel, -gcps[i].GCPLine))
    file.close()

def georef(input_file, reference_file, output_file,
           feature_name, target_resize_scale, reference_resize_scale,
           nfeat, flann, transform_type, resampleAlg,
           gcps_out, tile, offset,ki, kr, ratio=0.75,
           noData = 0, draw_plot=False, convert=False, apply_clahe=False):
    """Georeference the input using the training image and save the result in outputFile. 
    A ratio can be given to select more or less matches (defaults to 0.75)."""
    detector, norm = init_feature(feature_name, nfeat)
    
    logging.debug("%s : Detecting keypoints on target image", datetime.datetime.now())
    input_img_query, img_query, feature_name_query, kp_query, des_query = loadOrCompute(ki, input_file, feature_name, target_resize_scale, nfeat, tile, offset, convert, apply_clahe)
    
    logging.debug("%s : Detecting keypoints on reference image", datetime.datetime.now())
    input_img_train, img_train, feature_name_train, kp_train, des_train = loadOrCompute(kr, reference_file, feature_name, reference_resize_scale, nfeat, tile, offset, convert, apply_clahe)
    
    two_sides_matches = two_sides_match(norm, flann, des_train, des_query, ratio)

    MIN_MATCH_COUNT = 3

    projection, gcp_list, dst_gcps, matchesMask = getGCP(reference_file, kp_query, kp_train,
                                            two_sides_matches, min_matches=MIN_MATCH_COUNT)
    
    pointsShp = None
    pointsTxt = None
    if gcps_out:
        pointsShp = output_file[:-4] + '.shp'
        pointsTxt = output_file[:-4] + '.points'
    saveGeoref(input_file, output_file, projection, transform_type, resampleAlg,
               gcp_list, dst_gcps, pointsShp, pointsTxt, noData = 0)

    if draw_plot:
        draw_params = dict(#matchColor=(0, 255, 0),  # draw matches in green color
                           matchesThickness= 4,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=4)
        
        #Minmax histogram stretch for train image
        #equ_input_img_train = cv.equalizeHist(input_img_train)
        
        #Minmax histogram stretch for query image
        #equ_input_img_query = cv.equalizeHist(input_img_query)       
        
        #img3 = cv.drawMatches(equ_input_img_train, kp_train, equ_input_img_query, kp_query, two_sides_matches, None, **draw_params)
        img3 = cv.drawMatches(input_img_train, kp_train, input_img_query, kp_query, two_sides_matches, None, **draw_params)

        cv.imwrite(output_file[:-4] + '.png',img3)

def loadKeyPoints(points_file, n_features=0):
    with open(points_file, 'rb') as kpf:
        data = pickle.load(kpf)
        input_file = data['inputfile']
        feature_name = data['feature_name']
        logging.debug("%s : Loading keypoints extracted from %s using %s", datetime.datetime.now(), input_file,
                      feature_name)
        key_points = data['keypoints']
        logging.debug("%s : Found %d keypoints", datetime.datetime.now(), len(key_points))
        descriptors = data['descriptors']
        logging.debug("%s : Found %d descriptors", datetime.datetime.now(), len(descriptors))

        def make_cv_keypoint(kp):
            cvkp = cv.KeyPoint(x=kp[0], y=kp[1], _size=kp[2], _angle=kp[3], _response=kp[4], _octave=kp[5],
                               _class_id=kp[6])
            return cvkp

        cv_kp = [make_cv_keypoint(kp) for kp in key_points]
        # Sort the pairs of (keypoint, descriptor) based on keypoints responses, ascending
        sorted_array = sorted(zip(cv_kp, descriptors), key=lambda t: t[0].response, reverse=True)
        # Keep only nFeatures pairs is nFeatures is not 0
        sorted_array = sorted_array[:n_features] if n_features else sorted_array
        # Unzip so we get a list of sorted keypoints and a list of sorted descriptors
        unzipped = list(zip(*sorted_array))
        return input_file, feature_name, unzipped[0], unzipped[1]


def getGCP(reference_file, kp_query, kp_train, two_sides_matches, min_matches=3):
    if len(two_sides_matches) > min_matches:
        src_pts = np.float32([kp_train[m.queryIdx].pt for m in two_sides_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_query[m.trainIdx].pt for m in two_sides_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        matchesMask = mask.ravel().tolist()

        ds = gdal.Open(reference_file)
        x_offset, px_w, rot1, y_offset, px_h, rot2 = ds.GetGeoTransform()
        logging.debug("%s : Transform: %f, %f, %f, %f, %f, %f", datetime.datetime.now(), x_offset, px_w, rot1, y_offset,
                      px_h, rot2)

        gcp_list = []
        geo_t = ds.GetGeoTransform()
        # gcp_string = ''
        logging.debug("%s : %d matches", datetime.datetime.now(), len(matchesMask))
        for i, good_match in enumerate(matchesMask):
            if good_match == 1:
                p1 = kp_train[two_sides_matches[i].queryIdx].pt
                p2 = kp_query[two_sides_matches[i].trainIdx].pt
                pp = gdal.ApplyGeoTransform(geo_t, p1[0], p1[1])
                logging.debug("GCP geot = (%f,%f) -> (%f,%f)", p1[0], p1[1], pp[0], pp[1])
                logging.debug("Matched with (%f,%f)", p2[0], p2[1])
                z = 0
                # info = "GCP from pixel %f, %f" % (p1[0], p1[1])
                gcp = gdal.GCP(pp[0], pp[1], z, p2[0], p2[1])  # , info, i)
                # print ("GCP     = (" + str(p2[0]) +","+ str(p2[1]) + ") -> (" + str(pp[0]) +","+ str(pp[1]) + ")")
                # gcp_string += ' -gcp '+" ".join([str(p2[0]),str(p2[1]),str(pp[0]), str(pp[1])])
                gcp_list.append(gcp)
        logging.debug("%s : %d GCPs", datetime.datetime.now(), len(gcp_list))

        translate_t = gdal.GCPsToGeoTransform(gcp_list)
        translate_inv_t = gdal.InvGeoTransform(translate_t)
        logging.debug(len(translate_t))
        logging.debug("geotransform = %s", translate_t)
        logging.debug(len(translate_inv_t))
        logging.debug("invgeotransform = %s", translate_inv_t)
        # trans_gcp_list = []
        dst_gcp_list = []
        mapResiduals = 0.0
        geo_residuals = 0.0
        for gcp in gcp_list:
            # Inverse geotransform to get the corresponding pixel
            pix = gdal.ApplyGeoTransform(translate_inv_t, gcp.GCPX, gcp.GCPY)
            logging.debug("GCP = (%d,%d) -> (%d,%d)", gcp.GCPPixel, gcp.GCPLine, gcp.GCPX, gcp.GCPY)
            logging.debug(" => (%d,%d)", pix[0], pix[1])
            map_dX = gcp.GCPPixel - pix[0]
            map_dY = gcp.GCPLine - pix[1]
            map_residual = map_dX * map_dX + map_dY * map_dY
            mapResiduals = mapResiduals + map_residual
            # Apply the transform to get the GCP location in the output SRS
            pp = gdal.ApplyGeoTransform(translate_t, gcp.GCPPixel, gcp.GCPLine)
            z = 0
            out_gcp = gdal.GCP(pp[0], pp[1], z, gcp.GCPPixel, gcp.GCPLine)
            logging.debug("GCP = (%d,%d) -> (%d,%d)", out_gcp.GCPPixel, out_gcp.GCPLine, pp[0], pp[1])
            dX = gcp.GCPX - pp[0]
            dY = gcp.GCPY - pp[1]
            residual = dX * dX + dY * dY
            geo_residuals = geo_residuals + residual
            logging.debug("map residual = %f, %f = %f", map_dX, map_dY, map_residual)
            dst_gcp_list.append(out_gcp)

        logging.debug("map residuals: %s", mapResiduals)
        logging.debug("geo residuals: %s", geo_residuals)
        return ds.GetProjection(), gcp_list, dst_gcp_list, matchesMask
    else:
        logging.error("Not enough matches are found - %d/%d", len(two_sides_matches), min_matches)
        return None, None, None, None


def saveGeoref(input_file, output_file, projection, transform_type, resampleAlg,
               gcp_list, dst_gcp_list, points_shp = None, points_txt = None, noData = 0):
    src_ds = gdal.Open(input_file)
    # translate and warp the inputFile using GCPs and polynomial of order 1
    dst_ds = gdal.Translate('', src_ds, outputSRS=projection, GCPs=gcp_list, format='MEM')
    if transform_type == 'poly1':
        polynomialOrder = 1
        gdal.Warp(output_file, dst_ds, tps=False, polynomialOrder=polynomialOrder, dstNodata=noData, resampleAlg=resampleAlg)
    elif transform_type == 'poly2':
        polynomialOrder = 2
        gdal.Warp(output_file, dst_ds, tps=False, polynomialOrder=polynomialOrder, dstNodata=noData, resampleAlg=resampleAlg)
    elif transform_type == 'poly3':
        polynomialOrder = 3
        gdal.Warp(output_file, dst_ds, tps=False, polynomialOrder=polynomialOrder, dstNodata=noData, resampleAlg=resampleAlg)
    elif transform_type == 'tps':
        gdal.Warp(output_file, dst_ds, tps=True, dstNodata=noData, resampleAlg=resampleAlg)
    else:
        polynomialOrder = None
        gdal.Warp(output_file, dst_ds, tps=False, polynomialOrder=polynomialOrder, dstNodata=noData, resampleAlg=resampleAlg)
    # save the points to file
    if points_shp is not None:
        saveGCPsAsShapefile(dst_gcp_list, osr.SpatialReference(projection), points_shp)
    if points_txt is not None:
        saveGCPsAsText(dst_gcp_list, points_txt)
        
# ===================================================================================