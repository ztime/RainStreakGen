# Rain streak generator

Based on the [Photorealistic Rendering of Rain Streaks](http://www1.cs.columbia.edu/CAVE/publications/pdfs/Garg_TOG06.pdf)
and using implementation from [Yixin Juang](https://www.github.com/yxinjiang) and [Ruoteng Li](https://github.com/liruoteng). 

This implementation is very simplified though! It only generates rain streaks on solid black background

Some options and how to use it:
```
usage: generate_rain.py [-h] -o OUTPUT [-n NUMBER_OF_FRAMES] [-p PREFIX]
                        [-x DIMX] [-y DIMY] [--streak_folder STREAK_FOLDER]
                        [--img_channels IMG_CHANNELS]
                        [--intensity {dense,middle,light}]
                        [--angle {4,5,6,7,8}]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Folder to store results in
  -n NUMBER_OF_FRAMES, --number_of_frames NUMBER_OF_FRAMES
                        How many frames to generate
  -p PREFIX, --prefix PREFIX
                        Prefix of rendered images
  -x DIMX, --dimX DIMX
  -y DIMY, --dimY DIMY
  --streak_folder STREAK_FOLDER
                        Where the dataset of rain is stored
  --img_channels IMG_CHANNELS
                        How many channels to generate
  --intensity {dense,middle,light}
  --angle {4,5,6,7,8}
```

## Requirements
* Python 3.6+
* OpenCV
* skimage
* Pyplot

