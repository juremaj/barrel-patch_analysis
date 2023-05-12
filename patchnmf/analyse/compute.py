
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# image processing
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_li
from skimage.measure import find_contours

def downsample_tiff_avg(tiff, n=4):

    tiff_ds = []
    for i in range(tiff.shape[0]):
        kernel = np.ones((n, n))
        convolved = convolve2d(tiff[i,:,:], kernel, mode='valid')
        this_tiff_ds = convolved[::n, ::n] / n
        #tiff_ds = np.concatenate((tiff_ds, this_tiff_ds))
        tiff_ds.append(this_tiff_ds)
        
        if i % 100 == 0:
            print(f'Done with {i} frames') #to check progress

    tiff_ds = np.array(tiff_ds)

    plt.imshow(np.mean(tiff, 0), cmap='gray')
    plt.title('original frame example')
    plt.show()
    plt.imshow(np.mean(tiff_ds, 0), cmap='gray')
    plt.title('downsampled frame example')
    plt.show()

    return tiff_ds

def compute_nmfpx_blur_thr(nmf_px, tiff_shape, blur_std=6.5):
    _, x, y = tiff_shape
    n_components = nmf_px.n_components

    loading_imgs = []
    loading_imgs_filt = []
    rois_auto = []
    
    for i in range(n_components):

        loading = nmf_px.components_[i,:] # attr of nmf object
        loading_img = loading.reshape(x, y) #reshape ith nmf
        
        loading_img_filt, roi_auto = get_thr_img_auto(loading_img, blur_std, i) #blur and thresh
        
        # append to lists
        loading_imgs.append(loading_img)
        loading_imgs_filt.append(loading_img_filt)
        rois_auto.append(roi_auto) #add it to the list, fisrt iteration list is empty, after second iteration 1 nmf gets into teh lsit as threshloded matrix pf t and f
    
    return loading_imgs, loading_imgs_filt, rois_auto

def get_thr_img_auto(loading_img, blur_std, i):

    loading_img_filt = gaussian_filter(loading_img, blur_std)
    auto_thresh = threshold_li(loading_img_filt)
    roi_auto = loading_img_filt > auto_thresh
    
    return loading_img_filt, roi_auto

def get_roi_conts(rois_auto):
    conts = []
    n_conts = []

    for roi in rois_auto:
        roi_cont = find_contours(roi)
        conts.append(roi_cont)
        n_conts.append(len(roi_cont))

    
    return conts, n_conts

def get_loading_times(nmf_t):
    loading_times = []
    n_components = nmf_t.n_components

    for i in range(n_components):
        loading_time = nmf_t.components_[i, :]
        loading_times.append(loading_time)

    return loading_times