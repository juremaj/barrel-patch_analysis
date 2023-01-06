import os 
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def get_tiff(ds):
    tiff_all = []
    for (i, ti) in enumerate(os.listdir(f'data/{ds}/suite2p/plane0/reg_tif')):
        print(ti)
        if i == 0:
            tiff = io.imread(f'data/{ds}/suite2p/plane0/reg_tif/{ti}',  plugin='pil') # initialise tiff
        else:
            tiff_i = io.imread(f'data/{ds}/suite2p/plane0/reg_tif/{ti}',  plugin='pil')
            tiff = np.concatenate((tiff, tiff_i))
            print(tiff_i.shape)
        

    # making sure smallest value of tiff is zero - just a linear transform, shouldn't affect NMF
    tiff -= np.min(tiff)
    print(f'Shape of video: {tiff.shape}') 

    return tiff

def get_save_path(ds):
    save_path = os.getcwd() + '/data/' + ds + '/patch_sz/'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print('SavePath: ', save_path)
    return save_path

def export_conts_fiji(conts, save_path):
    # writing to text file (for FIJI export)
    for (i, roi_cont) in enumerate(conts):
        
        with open(save_path + f'roi_to_fiji/nmf{i+1}_roi.txt', 'w') as f:
            for j in range(len(roi_cont[0])):
                f.write(f'{roi_cont[0][j,1]}    {roi_cont[0][j,0]}\n')