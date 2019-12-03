import os
import sys
import argparse
import numpy as np
import cv2
import glob
from skimage.filters import gaussian
from skimage.util import img_as_float,img_as_uint
from matplotlib import pyplot as plt
from skimage.io import imsave
from pprint import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=os.path.abspath, help='Folder to store results in', required=True)
    parser.add_argument('-n', '--number_of_frames', type=int, default=100, help='How many frames to generate')
    parser.add_argument('-p', '--prefix', type=str, default='rain-frame', help="Prefix of rendered images")
    parser.add_argument('-x', '--dimX', type=int, default=128)
    parser.add_argument('-y', '--dimY', type=int, default=128)
    parser.add_argument('--streak_folder', type=os.path.abspath, default='data/Streaks_Garg06', help='Where the dataset of rain is stored')
    parser.add_argument('--img_channels', type=int, default=3, help='How many channels to generate')
    parser.add_argument('--intensity', type=str, choices=['dense', 'middle','light'], default='dense')
    parser.add_argument('--angle', type=int, default=4, choices=[4,5,6,7,8])
    args = parser.parse_args()
    # parser.add_argument('-p', '--prefix', help='Prefix for the files', required=True)
    # the last digit in the image '158-5.png' is an angle, which varies between 4 and 8.
    # thes means 10-4.png and 165-4.png are connected bc it's the same angle.
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    # Set intensity settings
    if args.intensity == 'dense':
        iterations = 8
    elif args.intensity == 'middle':
        iterations = 4
    elif args.intensity == 'light':
        iterations = 2

    all_frames = np.zeros((args.number_of_frames, args.dimX, args.dimY, args.img_channels))
    for frame_index in range(args.number_of_frames):
        available_images = glob.glob(f"{args.streak_folder}/*-{args.angle}.png")
        for _ in range(iterations): # We dont need to keep track






    # Algorithm:
    # For each image to add rain to:
        # Load file, get image,filename, height?, width?
            # for every angle X in 4,5,6,7,8:
                #copy image as floats between [0, 1] (no idea if its widthxheightx3 or how many channels)
                #create two zeroed images numpy arras w x h x 3
                #get a list of all images for angle X
                # for i in range 8: (this is probably bc we want DENSE) alter i depending on how intense
                # it should be
                    #Pick a random image for angle X and load it as n
                    #remove the first 2 rows? n = n[3:,:,:] n goes from (568 x 320 x 3?) -> (565,320, 3?)
                    # resize n into (565 * 4, 320 * 4) -> (2260, 1280)
                    # another zeros array same shape as n called new_n
                    # Convert n from RGB to grayscale BGR2GRAY -> img
                    # run imbinarize_0 on img -> bwst THIS IS SOMEKIND OF THRESHOLD! masky masky!
                    # create a mask bwareafilt(bwst) of any components that are connected
                    # but not to big
                    # For each color channel (3) as c:
                        #isolate the channel t = n[:,:,c]
                        # t = t * mask i.e only keep that which are not background ish
                        # apply some gaussian noise to the channel
                    # Resize back to original with and height
                    # 

                        
    

if __name__ == '__main__':
    main()
