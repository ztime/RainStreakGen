import numpy as np 
import os
import glob
import cv2
import random
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.util import img_as_float,img_as_uint
from skimage.io import imsave,imread

#Otsu's Binarization
def imbinarize_O(img, GaussianBlur=False):
    if GaussianBlur:
        img = cv2.GaussianBlur(img,(5,5),0)
    _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

#Adaptive Mean Thresholding
def imbinarize_M(img):
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th
#Adaptive Gaussian Thresholding
def imbinarize_G(img, GaussianBlur=False):
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th

def bwareafilt(image,min_size=0,max_size=1000,connectivity=4):
    image = image.astype(np.uint8)
    nb_components, labels , stats, _ = cv2.connectedComponentsWithStats(image, connectivity=connectivity)
    print(f"nb_components:{nb_components}")
    print(f"labels:{labels}")
    print(f"labels shape:{labels.shape}")
    print(f"stats:{stats}")
    print(f"stats shape:{stats.shape}")
    print(f"stats max: {np.amax(stats)}")
    sizes = stats[:, -1]
    print(f"sizes:{sizes}")
    print(f"sizes shape:{sizes.shape}")
    img2 = np.zeros(labels.shape)
    label = 1
    for i in range(2,nb_components):
        print(f"Running for {i}/{nb_components}...")
        if sizes[i] <= max_size and sizes[i]>=min_size:
            print(f"Size was ok!")
            label = i
            img2[labels == label] = 1
        fig = plt.figure()
        plt.imshow(img2)
        plt.show()
        plt.close(fig)
    return img2

def show_imgs_in_cols(rows,cols,img_list,title_list=[],figsize=(15, 10)):
    total = rows*cols
    fig = plt.figure(figsize=figsize)
    for i in range(min(len(img_list),total)):
        plt.subplot(rows,cols,i+1)
        plt.imshow(img_list[i])
        if len(title_list):
            plt.title(title_list[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
        
def write2txt(fname,strings,mode='a'):
    with open(fname,mode) as f:
        f.write(strings)

def add_rain_streak(im_path,str_index=4,stype='dense',gaussian_sigma = 1,st_min_size = 0,st_max_size = 1000 ,streak_root = 'data/Streaks_Garg06/'):  
    if stype=='dense':
        steps = 12
        iter_num = 0
        sigma = 0.2
        mu = 0.25
              
    elif stype == 'middle':
        steps = 4
        iter_num = 6
        sigma = 0.15
        mu = 0.2
           
    elif stype == 'sparse':
        steps = 2
        iter_num = 6
        sigma = 0.1
        mu = 0.1

    else:
        print("streak types only valid for 'dense','middle','sparse' ")
        return
    im = cv2.imread(im_path)
    filename = im_path.split(os.path.sep)[-1].split('.')[0]
    bh = im.shape[0]
    bw = im.shape[1]
    rain_final = img_as_float(im)
    stage_rain_final = img_as_float(im)
    st_final = np.zeros((bh,bw,3)).astype(float)
    str_file_list = glob.glob(streak_root+'*-{}.png'.format(str_index))
    stage_st_final = np.zeros((bh,bw,3)).astype(float)
    for i in range(steps):
        strnum = random.randint(1,len(str_file_list))
        st = cv2.imread(str_file_list[strnum])
        resize_shape = (int(st.shape[1]*4),int(st.shape[0]*4))
        st = cv2.resize(st,resize_shape)
        newst = np.zeros(st.shape).astype(float)
        img = cv2.cvtColor(st,cv2.COLOR_BGR2GRAY)
        bwst = imbinarize_O(img)
        mask = bwareafilt(bwst,st_min_size,st_max_size)
        for c in range(3):
            temp = st[:,:,c]
            temp = np.multiply(temp,mask.astype(np.uint8))
            newst[:,:,c] = gaussian(temp,gaussian_sigma,truncate=2)
        if not iter_num:     
            newst = cv2.resize(newst, (bw, bh))
            tr = random.random() * sigma + mu
            rain_final = rain_final + newst.astype(float) * tr
            stage_rain_final = stage_rain_final + newst.astype(float) * tr
            st_final = st_final + newst.astype(float)*tr
            stage_st_final = stage_st_final + newst.astype(float)*tr
        else:
            newst = cv2.resize(newst, (int(newst.shape[1]/2),int(newst.shape[0]/2)))
            sh = newst.shape[0] 
            sw = newst.shape[1]
            
            for iter in range(iter_num):
                row = random.randint(1,sh - bh)
                col = random.randint(1,sw - bw)
                selected = newst[row:row+bh, col:col+bw, :]
                tr = random.random() * sigma + mu
                rain_final = rain_final + selected.astype(float) * tr
                stage_rain_final = stage_rain_final + selected.astype(float) * tr
                st_final = st_final + selected.astype(float)*tr
                stage_st_final = stage_st_final + selected.astype(float)*tr
    rain_final = np.clip(rain_final,0.0,1.0)
    stage_rain_final = np.clip(stage_rain_final,0.0,1.0)
    pic_name = 'out/{0}-str{1}-type{2}.png'.format(stype,filename,str_index)
    imsave(pic_name,img_as_uint(stage_st_final))
    #write2txt(stype_list_stats,pic_name,'a')
    pic_name = 'out/{0}-rain{1}-type{2}.png'.format(stype,filename,str_index)
    imsave(pic_name,img_as_uint(rain_final))
    #write2txt(rain_stype_list_stats,pic_name,'a')
    return rain_final,st_final


    
if __name__ == "__main__":
    
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


    clean_list_stats = '%s/norain.txt'%filelist_name
    fileindex = 0
    im_path = img_file_list[fileindex]
    stypes = ['dense','middle','sparse']
    for stype in stypes:
        _,_=add_rain_streak(im_path,4,stype)

    '''
    im = cv2.imread(img_file_list[fileindex])
    filename = img_file_list[fileindex].split(os.path.sep)[-1].split('.')[0]
    bh = im.shape[0]
    bw = im.shape[1]
    str_index = 4
    clean_final = img_as_float(im)
    st_final = np.zeros((bh,bw,3)).astype(float)
    str_file_list = glob.glob(streak_root+'*-{}.png'.format(str_index))
    stage_st_final = np.zeros((bh,bw,3)).astype(float)
    #dense
    for i in range(8):
        strnum = random.randint(1,len(str_file_list))
        st = cv2.imread(str_file_list[strnum])
        #st = st[3:,:,:]

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

    st_org = cv2.resize(cv2.imread(str_file_list[strnum]),(bw,bh))        
    images = [st_org,clean_final,st_final,stage_st_final]
    fig = plt.figure(figsize=(20, 12))
    
    titles = ["st","clean_final","st_final","stage_st_final"]
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()        
    pic_name = 'out/str{0}-type{1}-dense.png'.format(filename,str_index)
    imsave(pic_name,img_as_uint(stage_st_final))
    write2txt(dense_list_stats,pic_name,'a')

    #st_final = np.zeros((bh,bw,3)).astype(float)
    stage_st_final = np.zeros((bh,bw,3)).astype(float)
    #clean_final = img_as_float(im)
    for i in range(8):
        strnum = random.randint(1,len(str_file_list))
        st = cv2.imread(str_file_list[strnum])
        #st = st[3:,:,:]
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
            row = random.randint(1,sh - bh)
            col = random.randint(1,sw - bw)
            selected = newst[row:row+bh, col:col+bw, :]
            tr = random.random() * 0.15 + 0.20
            clean_final = clean_final + selected.astype(float) * tr
            st_final = st_final + selected.astype(float)*tr
            stage_st_final = stage_st_final + selected.astype(float)*tr
    st_org = cv2.resize(cv2.imread(str_file_list[strnum]),(bw,bh))        
    images = [st_org,clean_final,st_final,stage_st_final]
    fig = plt.figure(figsize=(20, 12))
    
    titles = ["st","clean_final","st_final","stage_st_final"]
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()        
        #write middle streak
    pic_name = 'out/str{0}-type{1}-mid.png'.format(filename,str_index)
    imsave(pic_name,img_as_uint(stage_st_final))       
    # write middle streak file into file list
    write2txt(middle_list_stats,pic_name,'a')
    '''


