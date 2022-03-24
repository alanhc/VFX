from unicodedata import name
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import os
import sys
from scipy.linalg import lstsq

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", help="Path to the input images", type=str, required=True)
parser.add_argument("--out", help="Output name", type=str, required=True)
parser.add_argument("--l", help="Weight of smoothness", type=int, default=100)
parser.add_argument("--w", help="Weight function for pixel values",
                    type=str, choices=["binomial", "uniform", "distance"], default="binomial")
parser.add_argument("--origin", action="store_true")
args = parser.parse_args()

input_path = os.path.join(args.input, "*JPG")
output_path = os.path.join("./result", args.out)

try:
    os.mkdir("./result")
except:
    pass

try:
    os.mkdir(output_path)
except:
    pass

files = glob.glob(input_path)
files.sort(reverse=False)
if args.origin:
    raw_images = [cv2.imread(f) for f in files]
else:
    raw_images = [cv2.resize(cv2.imread(f), (1440, 960),
                             interpolation=cv2.INTER_AREA) for f in files]
images = [image[32:-24, 16:-16] for image in raw_images]

shutter_speed = np.array(
    [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

plt.figure(figsize=(12, 12))
for i in range(len(images)):
    plt.subplot(4, math.ceil(len(images)/4), i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
plt.savefig(os.path.join(output_path, "inputs.png"))


def get_sample():
    sample = []
    for i in range(len(images)):
        sample.append(cv2.resize(
            images[i], (10, 10), interpolation=cv2.INTER_AREA))
    sample = np.array(sample)

    sample = np.transpose(sample, (3, 1, 2, 0))
    sample = sample.reshape(
        (sample.shape[0], sample.shape[1]*sample.shape[2], sample.shape[3]))
    return sample

# gsolve(Z, B, l, w)


def gsolve(Z, B, l, w):

    n = 256
    A = np.zeros((Z.shape[0]*Z.shape[1]+(n-2)+1,
                 n+Z.shape[0]), dtype=np.float32)
    b = np.zeros((A.shape[0], 1), dtype=np.float32)
    W = np.zeros(Z.shape, dtype=np.float32)

    # 算np
    k = 0
    for i in range(0, Z.shape[0]):
        for j in range(0, Z.shape[1]):
            W[i][j] = w(Z[i][j])
            A[k][Z[i][j]] = W[i][j]
            A[k][n+i] = -W[i][j]
            b[k][0] = W[i][j] * B[j]
            k += 1

    A[k][127] = 1
    k += 1

    for i in range(0, n-2):
        weight = w(i+1)
        A[k][i] = l * weight
        A[k][i+1] = -2 * l*weight
        A[k][i+2] = l * weight
        k += 1

    x = lstsq(A, b)

    g = x[0][:n]
    lE = x[0][n:]
    return g, lE


def draw_response_curve(gs):
    colors = ["blue", "green", "red"]
    plt.figure(figsize=(6, 4))
    for i in range(3):
        plt.plot(range(256), gs[i], color=colors[i])
    plt.xlabel("z")
    plt.ylabel("g(z)")
    plt.savefig(os.path.join(output_path, "response_curve.png"))


def construct_radiance_map(images, B, gs, w):
    w_map = [w(x) for x in range(256)]
    shape = images[0].shape
    images = np.array(images)
    weights = np.vectorize(lambda x: w_map[x])(images)
    w_t = np.sum(weights, axis=0)

    def get_g(x):
        a = np.zeros(3)
        a[0] = gs[0][x[0]]
        a[1] = gs[1][x[1]]
        a[2] = gs[2][x[2]]
        return a
    lnE_raw = np.apply_along_axis(
        get_g, 3, images) - np.array([np.ones(shape) * t for t in B])
    sum = np.sum(weights * lnE_raw, axis=0)
    return sum / w_t


# Weight functions
mu = np.mean(range(256))
sigma = np.std(range(256))

# Binomail distribution pdf


def binomial(x): return (1 / (sigma * (2 * np.pi)**0.5)) * \
    np.exp((-1/2) * (((x - mu) / sigma)**2))
# uniform
def uniform(x): return x
# 1/Distance
def distance(x): return mu - abs(x - mu)


shutter_time = 1/shutter_speed
B = np.log(shutter_time)
l = args.l
w = uniform if (args.w == "uniform") else distance if (
    args.w == "distance") else binomial

sample = get_sample()

gs, lEs = [], []
for i in range(3):
    Z = sample[i]
    g, lE = gsolve(Z, B, l, w)
    gs.append(g)
    lEs.append(lE)

draw_response_curve(gs)

ln_radiance_map = construct_radiance_map(raw_images, B, gs, w)

for i in range(3):
    plt.imshow(ln_radiance_map[:, :, i], cmap="jet")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "radiance_map{0}.png".format(i)))
    plt.clf()

radiance_map = np.exp(ln_radiance_map)
cv2.imwrite(os.path.join(output_path, args.out + '.hdr'),
            radiance_map.astype(np.float32))
