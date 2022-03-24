import cv2
import glob
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

path = "./data/" + sys.argv[1] + "/*JPG"
files = glob.glob(path)
files.sort()
# images = [cv2.resize(cv2.imread(f), (200, 300),
#                      interpolation=cv2.INTER_AREA) for f in files]
images = [cv2.imread(f) for f in files]
shutter_speed = np.array(
    [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
try:
    os.mkdir(os.path.join("./output/", sys.argv[1]))
except:
    print("Directory exists.")
os.popen("cp " + files[7] + " " + os.path.join("./output/",
         sys.argv[1], os.path.basename(files[7])))


def ImageShrink2(img):
    """
    shrink image by 2
    """
    (h, w) = img.shape[:2]
    return cv2.resize(img, (w//2, h//2))


def ComputeBitmaps(img):
    """
    compute median threshold bitmap and exclusion bitmap
    """
    # m = int(np.median(img))
    # threshold_bitmap = cv2.threshold(img, m, 255, cv2.THRESH_BINARY)[1]

    # exclusion_bitmap = 255 - cv2.inRange(img, m - 2, m + 2)

    threshold = 2
    median = int(np.median(img))
    threshold_bitmap = np.where(img > median, 255, 0).astype(np.uint8)
    exclusion_bitmap = np.where(
        np.abs(img - median) <= threshold, 0, 255).astype(np.uint8)
    return threshold_bitmap, exclusion_bitmap
    # med = int(np.median(img))
    # thresBitmap = np.array(
    #     [[True if yi > med else False for yi in xi] for xi in img], dtype='bool')
    # x, y = img.shape
    # excluBitmap = np.full((x, y), True, dtype='bool')
    # for i in range(x):
    #     for j in range(y):
    #         if abs(img[i][j] - med) < 5:
    #             excluBitmap[i][j] = False

    # return thresBitmap, excluBitmap


def BitmapShift(bm, x, y):
    """
    shift image by x, y
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    shape = bm.shape
    return cv2.warpAffine(bm, M, (shape[1], shape[0]))


# ref: [1] - Overall algorithm
def getExpShift(img1, img2, shiftBits):
    if (shiftBits > 0):
        sml_img1 = ImageShrink2(img1)
        sml_img2 = ImageShrink2(img2)
        cur_shift = getExpShift(sml_img1, sml_img2, shiftBits-1)
        cur_shift[0] *= 2
        cur_shift[1] *= 2
    else:
        cur_shift = [0, 0]
    tb1, eb1 = ComputeBitmaps(img1)
    tb2, eb2 = ComputeBitmaps(img2)
    min_err = img1.shape[0] * img1.shape[1]
    shift_ret = [0, 0]
    for x in range(-1, 2):
        for y in range(-1, 2):
            xs = cur_shift[0] + x
            ys = cur_shift[1] + y
            shifted_tb2 = BitmapShift(tb2, xs, ys)
            shifted_eb2 = BitmapShift(eb2, xs, ys)
            diff_b = np.logical_xor(tb1, shifted_tb2)
            diff_b = np.logical_and(diff_b, eb1)
            diff_b = np.logical_and(diff_b, shifted_eb2)
            err = np.sum(diff_b)
            if (err < min_err):
                shift_ret = [xs, ys]
                min_err = err
    return shift_ret


level = 2
image_shifted = [0] * len(images)
g1 = cv2.cvtColor(images[7], cv2.COLOR_BGR2GRAY)
for i in range(0, len(images)):
    if i == 7:
        continue
    g2 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    x, y = getExpShift(g1, g2, level)
    image_shifted[i] = BitmapShift(images[i], x, y)
    # print("out/"+os.path.basename(files[i]))
    cv2.imwrite(os.path.join(
        "./output/", sys.argv[1], os.path.basename(files[i])), image_shifted[i])
    print(x, y)
