import logging
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from load_keypoints import loadKeyPoints
from common import init_feature, two_sides_match
import datetime
import click

# Configure the global logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

@click.option('-i', type=click.File('rb', lazy=True), help='Input keypoints')
@click.option('-r', type=click.File('rb', lazy=True), help='Reference keypoints')
@click.option('-flann/-no-flann', 'flann', default=False, help='use flann matcher')
@click.option('-gcps/-no-gcps', 'gcps', default=False, help='export gcps')
@click.option('-ratio', type=float, help='ratio test explained by D.Lowe')

def match_keypoints(i,r,flann,gcps,ratio):
    train, feature_name_train, kp_train, des_train = loadKeyPoints(r)
    query, feature_name_query, kp_query, des_query = loadKeyPoints(i)
    assert feature_name_train == feature_name_query
    _, norm = init_feature(feature_name_query)
    ratio = ratio
    if ratio is None:
        ratio = 0.75
    
    # Match both ways
    two_sides_matches = two_sides_match(norm, flann, des_train, des_query, ratio)

    MIN_MATCH_COUNT = 3

    matchesMask = None
    if len(two_sides_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_train[m.queryIdx].pt for m in two_sides_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_query[m.trainIdx].pt for m in two_sides_matches]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        matchesMask = mask.ravel().tolist()
        logging.debug("%s : Matches Mask %d", datetime.datetime.now(), len(matchesMask))

    img1 = cv.imread(train)
    img2 = cv.imread(query)
    #img3 = cv.drawMatchesKnn(img1,kp_train,img2,kp_query,two_sides_matches,outImg=None, flags=2)#None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1,kp_train,img2,kp_query, two_sides_matches,
               outImg=img_match, matchColor=None, singlePointColor=None, matchesMask = matchesMask, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB)),plt.show()
    
if __name__ == "__main__":
    match_keypoints()