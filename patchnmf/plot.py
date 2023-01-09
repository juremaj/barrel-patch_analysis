import numpy as np
import matplotlib.pyplot as plt

def plot_nmfpx_blur_thr(loading_imgs, loading_imgs_filt, rois_auto):
    n_components = len(loading_imgs)

    all_im_list = [loading_imgs, loading_imgs_filt, rois_auto]

    fig, axs = plt.subplots(n_components, 3, figsize=(9, 3*n_components))

    for (i, im_list) in enumerate(all_im_list): # why does this need to be indented????
        for (j, im) in enumerate(im_list):
            axs[j,i].imshow(im, cmap='gray')


            axs[j,i].xaxis.set_ticklabels('') 
            axs[j,i].yaxis.set_ticklabels('') 
            axs[j,i].xaxis.set_ticks([]) 
            axs[j,i].yaxis.set_ticks([]) 

            if i == 0:
                axs[j,i].set_ylabel(f'NMF {j}')

            if j == 0:
                axs[j,0].set_title(f'Raw pxNMF')
                axs[j,1].set_title(f'Smoothed pxNMF')
                axs[j,2].set_title(f'Sm. and thr. pxNMF')


def plot_nmf_t(nmf_t, gt_acts=None, plot_gt=False):

    fig, axs = plt.subplots(nmf_t.components_.shape[0], 1, figsize=(10, nmf_t.components_.shape[0]))

    for i in range(nmf_t.components_.shape[0]):
        axs[i].plot(nmf_t.components_[i, :]/np.max(nmf_t.components_[i, :]), c='C1')
        if plot_gt:
            axs[i].plot(gt_acts[i]/np.max(gt_acts[i]), c='C0')
            
        axs[i].axis('off')

    plt.show()

def plot_rois_overlay(rois_auto, tiff_shape):

    n, x, y = tiff_shape
    plt.figure(figsize=(10,10))

    plt.figure(figsize=(10,10))

    for (i, roi) in enumerate(reversed(rois_auto)): # reversed to plot more obvious components first
        roi_scat = np.nonzero(roi)
        plt.scatter(roi_scat[1], -roi_scat[0], marker='s', s=9, alpha=0.1) # - is because of image processing convention
        plt.ylim((-y, 0))
        plt.xlim((0,x))
        plt.axis('off')
    plt.show()

def plot_roi_conts_largest(conts, tiff_shape):
    
    # plots the largest contour in each ROI
    
    n, x, y = tiff_shape

    plt.figure(figsize=(10,10))

    for (i, roi_cont) in enumerate(conts):

        plt.plot(roi_cont[0][:,1], roi_cont[0][:,0], linewidth=5, alpha=0.7)

        plt.ylim((y, 0))
        plt.xlim((0,x))
        plt.axis('off')

def plot_roi_area_hist(rois_auto, n_bins=10, resolution=1.2):
    roi_areas = [np.sum(roi)*(resolution**2) for roi in rois_auto]
    plt.hist(roi_areas, n_bins);
    plt.title('patch area distribution')
    plt.xlabel('area (um^2)')

def plot_px_nmf_corr(nmf_px):
    plt.figure(figsize=(3,3), dpi=200)
    nmf_px_corrmat = np.corrcoef(nmf_px.components_) # correlation matrix of this
    plt.imshow(nmf_px_corrmat, vmin=0, vmax=1)
    plt.colorbar() 
    plt.xlabel('NMF component')
    plt.ylabel('NMF component')
    plt.xticks([])
    plt.yticks([])

def plot_roi_loading_time(rois_auto, loading_times, title='NOTE: the L and R do not necc. correspond'):
    n_components = len(rois_auto)

    fig, axs = plt.subplots(n_components, 2, figsize=(5, n_components), width_ratios=[1, 5], dpi=200)
    plt.suptitle(title, fontsize=10)

    for (i, loading_time) in enumerate(loading_times):
        axs[i,0].imshow(rois_auto[i], cmap='gray')
        axs[i,0].xaxis.set_ticklabels('') 
        axs[i,0].yaxis.set_ticklabels('') 
        axs[i,0].xaxis.set_ticks([]) 
        axs[i,0].yaxis.set_ticks([]) 

        axs[i,1].plot(loading_time, c='grey')
        axs[i,1].axis('off')

        if i == 0:
            axs[i,0].set_title('PX component', fontsize=7)
            axs[i,1].set_title('T component (activation of PX component over time)', fontsize=7)
    
    plt.show()