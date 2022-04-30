import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
from tools import warp, harris, get_keypoints_and_orientations, SIFT_descriptor, match

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to the input images", type=str, required=True)
parser.add_argument("--focal", help="Weight of smoothness", type=int, default=705)
parser.add_argument("--cache", help="cache descriptor", type=bool, default=False)
args = parser.parse_args()


focal_len = args.focal
path = "./data/"+args.input
files = glob.glob(path)
files.sort(reverse=False)
images = [cv2.imread(f) for f in files][:2]
# warp images
img_warp = [warp(img, focal_len) for img in images]



gray_images, harris_images, key_points_all, orientations = get_keypoints_and_orientations(img_warp, 3, 3, 0.05, 1) # img_warp, ksize, gksize, k, threshold

descriptors = []
if (args.cache):
    log_files = glob.glob("log/descriptor-*.npy")
    for f in log_files:
        descriptor = np.load(f)
        descriptors.append(descriptor)
    print("using cached descriptors")
else:
    for i in range(len(gray_images)):
        print("descriptor for image {}".format(i))
        descriptor = SIFT_descriptor(gray_images[i], key_points_all[i], orientations[i])
        with open('log/descriptor-{}.npy'.format(str(i)), 'wb') as f:
            np.save(f, descriptor)
        descriptors.append( descriptor )
    
matches = match(descriptors)
print(matches)