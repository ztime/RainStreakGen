import os
import sys
import argparse
import numpy as np
import cv2
import random
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
        print(f"Working on {frame_index}/{args.number_of_frames}...")
        available_images = glob.glob(f"{args.streak_folder}/*-{args.angle}.png")
        for _ in range(iterations): # We dont need to keep track
            random_streak_img = random.choice(available_images)
            random_streak_img = cv2.imread(random_streak_img)
            # Crop 3 pixel rows to get a picture that can scale evenly
            random_streak_img = random_streak_img[3:, :, :]
            # Flip the width and height here, not sure why
            resize_shape = random_streak_img.shape
            resize_shape = (resize_shape[1] * 4, resize_shape[0] * 4)
            random_streak_img_resized = cv2.resize(random_streak_img, resize_shape)
            # put filtered channels in a new frame
            new_frame = np.zeros_like(random_streak_img_resized, dtype=np.float)
            thresholded_frame = imbinarize_O(cv2.cvtColor(random_streak_img_resized, cv2.COLOR_BGR2GRAY))
            frame_mask = bwareafilt(thresholded_frame)
            for img_channel in range(3):
                one_channel = random_streak_img_resized[:,:,img_channel]
                one_channel = np.multiply(one_channel, frame_mask.astype(np.uint8))
                # No idea about these settings, from the original
                one_channel = gaussian(one_channel, 1, truncate=2)
                new_frame[:,:,img_channel] = one_channel
            # Take the size down again
            new_frame = cv2.resize(new_frame, (args.dimX, args.dimY))
            # Add it to the frame with a few modifications
            alpha = random.random() * 0.2 + 0.25
            filtered_frame = new_frame * alpha
            all_frames[frame_index] = all_frames[frame_index] + (new_frame.astype(float) * alpha)
        frame_prepend_zeros = len(str(args.number_of_frames))
        filename = os.path.join(args.output, f"{args.prefix}-{frame_index:0>{frame_prepend_zeros}}.png")
        imsave(filename, img_as_uint(all_frames[frame_index]))

    print(f"Done with {args.number_of_frames} frames!")

def bwareafilt(image,min_size=0,max_size=1000,connectivity=4):
    image = image.astype(np.uint8)
    nb_components, labels , stats, _ = cv2.connectedComponentsWithStats(image, connectivity=connectivity)
    sizes = stats[:, -1]
    img2 = np.zeros(labels.shape)
    label = 1
    for i in range(2,nb_components):
        if sizes[i] <= max_size and sizes[i]>=min_size:
            label = i
            img2[labels == label] = 1
    return img2

#Otsu's Binarization
def imbinarize_O(img, GaussianBlur=False):
    if GaussianBlur:
        img = cv2.GaussianBlur(img,(5,5),0)
    _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th


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
