import numpy as np 
import os
import glob
import cv2
import random
from matplotlib import pyplot as plt

#Otsu's Binarization
def imbinarize_O(img, GaussianBlur=False):
    if GaussianBlur:
        img = cv2.GaussianBlur(img,(5,5),0)
    _,th = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

#Adaptive Mean Thresholding
def imbinarize_M(img):
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th
#Adaptive Gaussian Thresholding
def imbinarize_G(img, GaussianBlur=False):
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th

def bwareafilt(image,min_size=10,max_size=1000,connectivity=4):
    image = image.astype(np.uint8)
    nb_components, labels , stats, _ = cv2.connectedComponentsWithStats(image, connectivity=connectivity)
    sizes = stats[:, -1]
    img2 = np.zeros(labels.shape)
    label = 1
    for i in range(2, nb_components):
        if sizes[i] <= max_size and sizes[i]>=min_size:
            label = i
            img2[labels == label] = 255

    return img2

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

fileindex = 0
im = cv2.imread(img_file_list[fileindex])
filename = img_file_list[fileindex].split(os.path.sep)[-1]
bh = im.shape[0]
bw = im.shape[1]

clean_final = im.astype(float)
st_final = np.zeros((bh,bw,3)).astype(float)
str_file_list = glob.glob(streak_root+'*.png')
stage_st_final = np.zeros((bh,bw,3)).astype(float)
        #dense
strnum = 0 #random.randint(1,len(str_file_list))
st = cv2.imread(str_file_list[strnum])
#st = st[3:,:,:]
#resize_shape = (int(st.shape[0]*4),int(st.shape[1]*4))
#st = cv2.resize(st,resize_shape)
newst = np.zeros(st.shape).astype(float)
img = cv2.cvtColor(st,cv2.COLOR_BGR2GRAY)
bwst = imbinarize_O(img)
mask = bwareafilt(bwst)
images = [st,bwst,mask]
fig = plt.figure(figsize=(20, 12))
titles = ["Original","Otsu's Binarization","Otsu's Binarization with GaussianBlur"]
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

