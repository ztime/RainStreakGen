import os
import sys
import glob
import argparse
import numpy as np
import cv2
from skimage.util import img_as_float,img_as_uint
from matplotlib import pyplot as plt
from skimage.io import imsave
from pprint import pprint

def main():
    print("hej")
    files = sorted(glob.glob('rain_light_4/*.png'))
    frames = np.zeros((len(files) - 1, 128, 128))
    for i in range(len(files) - 1):
        img_a = cv2.imread(files[i])
        img_b = cv2.imread(files[i + 1])
        diff = img_a - img_b
        diff2 = img_b - img_a
        diff3 = np.abs(img_a - img_b)
        filterd = np.copy(diff3)
        filterd[filterd > np.average(filterd)] = 128
        filterd[filterd <= np.average(filterd)] = 0
        fix, axes = plt.subplots(2,4)
        axes[0, 0].imshow(img_a)
        axes[0, 1].imshow(img_b)
        axes[0, 2].imshow(diff)
        axes[0, 3].imshow(diff2)
        axes[1, 0].imshow(diff3)
        axes[1, 1].imshow(filterd)
        plt.show()
        print(np.amax(img_a.shape))
        print(np.amin(img_a.shape))
        print(np.average(img_a.shape))
        quit()

        


if __name__ == '__main__':
    main()
