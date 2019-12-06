import os
import sys
import glob
import argparse
import numpy as np
import cv2
from skimage.util import img_as_float,img_as_uint
from matplotlib import pyplot as plt
from matplotlib import animation
from skimage.io import imsave
from pprint import pprint

def main():
    print("hej")
    files = sorted(glob.glob('rain_light_4/*.png'))
    frames = np.zeros((len(files) - 2, 128, 128))
    all_frames = np.array([cv2.imread(f) for f in files])
    print(all_frames.shape)
    average_pixel_values = np.average(all_frames, axis=0)
    fig = plt.figure()
    ani = []
    for i in range(len(files) - 2):
        img_a = cv2.imread(files[i])
        img_b = cv2.imread(files[i + 1])
        img_c = cv2.imread(files[i + 2])
        mask_a = (img_a > average_pixel_values) & (img_b > average_pixel_values) & (img_c > average_pixel_values)
        all_frames[i] = mask_a.astype(np.float)
        image = plt.imshow(mask_a.astype(np.float))
        ani.append([image])
        # mask_b = img_b > average_pixel_values
        # print(mask_a.shape)
        # print(np.amax(mask_a))
        # diff = img_a - img_b
        # diff2 = img_b - img_a
        # diff3 = np.abs(img_a - img_b)
        # filterd = np.copy(diff3)
        # filterd[filterd > np.average(filterd)] = 128
        # filterd[filterd <= np.average(filterd)] = 0
        # fix, axes = plt.subplots(2,4)
        # axes[0, 0].imshow(img_a)
        # axes[0, 1].imshow(img_b)
        # axes[0, 2].imshow(diff)
        # axes[0, 3].imshow(diff2)
        # axes[1, 0].imshow(diff3)
        # axes[1, 1].imshow(filterd)
        # axes[1, 2].imshow(average_pixel_values)
        # axes[1, 3].imshow(mask_a.astype(np.float))
        # plt.tight_layout()
        # plt.show()
        # print(np.amax(img_a.shape))
        # print(np.amin(img_a.shape))
        # print(np.average(img_a.shape))
        # quit()

    # anim = animation.ArtistAnimation(fig, ani, interval=50, blit=True, repeat_delay=1000)
    # anim = animation.ArtistAnimation(fig, ani, interval=50, blit=True)
    anim = animation.ArtistAnimation(fig, ani, interval=30, blit=True)
    plt.show()

if __name__ == '__main__':
    main()
