import numpy as np 
import os
import glob
import cv2
import random
from utils import imbinarize_O,bwareafilt,write2txt
from skimage.filters import gaussian
from skimage.util import img_as_float,img_as_uint
from matplotlib import pyplot as plt
from skimage.io import imsave
from pprint import pprint


streak_root = 'data/Streaks_Garg06/'
image_root = 'data/BSD300/'
img_file_list = glob.glob(image_root+'*.jpg')
num_of_strtype = 5
num_of_file = len(img_file_list)
filelist_name = 'filelist_new'
cwd = os.getcwd()

out_path = 'new_out'

print(f"Files to do?:{img_file_list}")

if not os.path.exists(os.path.join(cwd,out_path)):
    os.mkdir(os.path.join(cwd,out_path))
if not os.path.exists(os.path.join(cwd,filelist_name)):
    os.mkdir(os.path.join(cwd,filelist_name))

rain_list_stats = '%s/rain.txt'%filelist_name
sparse_list_stats = '%s/sparse.txt'%filelist_name
middle_list_stats = '%s/mid.txt'%filelist_name
dense_list_stats = '%s/dense.txt'%filelist_name
clean_list_stats = '%s/norain.txt'%filelist_name

for fileindex in range(num_of_file):
    print(f"File {fileindex}:{img_file_list[fileindex]}")
    im = cv2.imread(img_file_list[fileindex])
    filename = img_file_list[fileindex].split(os.path.sep)[-1].split('.')[0]
    bh = im.shape[0]
    bw = im.shape[1]


    for str_index in range(4,9):
        clean_final = img_as_float(im)
        st_final = np.zeros((bh,bw,3)).astype(float)
        str_file_list = glob.glob(streak_root+'*-{}.png'.format(str_index))
        print('---------------------------------------')
        print(str_file_list)
        stage_st_final = np.zeros((bh,bw,3)).astype(float)
        #dense
        for i in range(8):
            strnum = random.randint(0,len(str_file_list) - 1)
            st = cv2.imread(str_file_list[strnum])
            st = st[3:,:,:]
            resize_shape = (int(st.shape[1]*4),int(st.shape[0]*4))
            st = cv2.resize(st,resize_shape)
            newst = np.zeros(st.shape).astype(float)
            img = cv2.cvtColor(st,cv2.COLOR_BGR2GRAY)
            bwst = imbinarize_O(img)
            mask = bwareafilt(bwst)
            for c in range(3):
                temp = st[:,:,c]
                temp = np.multiply(temp,mask.astype(np.uint8))
                newst[:,:,c] = gaussian(temp,1,truncate=2)
                    
            newst = cv2.resize(newst, (bw, bh))
            tr = random.random() * 0.2 + 0.25
            clean_final = clean_final + newst.astype(float) * tr
            st_final = st_final + newst.astype(float)*tr
            stage_st_final = stage_st_final + newst.astype(float)*tr

        # write dense streak
        pic_name = '{2}/str{0}-type{1}-dense.png'.format(filename,str_index,out_path)
        imsave(pic_name,img_as_uint(stage_st_final))       
        # write dense streak file into file list
        print(f"Writing {pic_name}...")
        write2txt(dense_list_stats,pic_name,'a')

        # middle
        stage_st_final = np.zeros((bh,bw,3)).astype(float)
        for i in range(2):
            strnum = random.randint(1,len(str_file_list) - 1)
            st = cv2.imread(str_file_list[strnum])
            st = st[3:,:,:]
            resize_shape = (int(st.shape[1]*4),int(st.shape[0]*4))
            st = cv2.resize(st,resize_shape) 
            newst = np.zeros(st.shape).astype(float)
            img = cv2.cvtColor(st,cv2.COLOR_BGR2GRAY)
            bwst = imbinarize_O(img)
            mask = bwareafilt(bwst,800,4000)

            for c in range(3):
                temp = st[:,:,c]
                temp = np.multiply(temp,mask.astype(np.uint8))
                newst[:,:,c] = gaussian(temp,2,truncate=2)
            
            newst = cv2.resize(newst, (int(newst.shape[1]/2),int(newst.shape[0]/2)))
            sh = newst.shape[0] 
            sw = newst.shape[1]
            
            for iter in range(6):
                row = random.randint(0,sh - bh)
                col = random.randint(0,sw - bw)
                selected = newst[row:row+bh, col:col+bw, :]
                tr = random.random() * 0.15 + 0.20
                clean_final = clean_final + selected.astype(float) * tr
                st_final = st_final + selected.astype(float)*tr
                stage_st_final = stage_st_final + selected.astype(float)*tr
            
            
        #write middle streak
        pic_name = '{2}/str{0}-type{1}-mid.png'.format(filename,str_index,out_path)
        print(f"Writing {pic_name}...")
        imsave(pic_name,img_as_uint(stage_st_final))       
        # write middle streak file into file list
        write2txt(middle_list_stats,pic_name,'a')




