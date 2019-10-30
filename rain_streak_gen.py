import numpy as np 
import os
import glob
import cv2
import random
from utils import imbinarize_O,bwareafilt
import skimage.filters.gaussian_filter as gaussian_filter

streak_root = 'data/Streaks_Garg06/'
image_root = 'data/BSD300/'
img_file_list = glob.glob(image_root+'*.jpg')
num_of_strtype = 5
num_of_file = len(img_file_list)
filelist_name = 'filelist'
cwd = os.getcwd()

if not os.path.exists(os.path.join(cwd,'out')):
    os.mkdir(os.path.join(cwd,'out'))
if not os.path.exists(os.path.join(cwd,filelist_name)):
    os.mkdir(os.path.join(cwd,filelist_name))

rain_list_stats = open('%s/rain.txt'%filelist_name,'w')
sparse_list_stats = open('%s/sparse.txt'%filelist_name,'w')
middle_list_stats = open('%s/mid.txt'%filelist_name,'w')
dense_list_stats = open('%s/dense.txt'%filelist_name,'w')
clean_list_stats = open('%s/norain.txt'%filelist_name,'w')

for fileindex in range(num_of_file):
    im = cv2.imread(img_file_list[fileindex])
    filename = img_file_list[fileindex].split(os.path.sep)[-1]
    bh = im.shape[0]
    bw = im.shape[1]

    for str_index in range(1):
        clean_final = im.astype(float)
        st_final = np.zeros((bh,bw,3)).astype(float)
        str_file_list = glob.glob(streak_root+'*.png')
        stage_st_final = np.zeros((bh,bw,3)).astype(float)
        #dense
        for i in range(8):
            strnum = random.randint(1,len(str_file_list))
            st = cv2.imread(str_file_list[strnum])
            st = st[3:,:,:]
            resize_shape = (int(st.shape[0]*4),int(st.shape[1]*4))
            st = cv2.resize(st,resize_shape)
            newst = np.zeros(st.shape).astype(float)
            img = cv2.cvtColor(st,cv2.COLOR_BGR2GRAY)
            bwst = imbinarize_O(img)
            mask = bwareafilt(bwst)

            for c in range(3):
                temp = st[:,:,c]
                temp = np.dot(temp,mask.astype(np.uint8))
                newst[:,:,c] = gaussian_filter(im, 1, multichannel=True, mode='reflect')
        
            
            newst = cv2.imresize(newst, (bh, bw))
            tr = random.random() * 0.2 + 0.25
            clean_final = clean_final + newst.astype(float) * tr
            st_final = st_final + newst.astype(float)*tr
            stage_st_final = stage_st_final + newst.astype(float)*tr






