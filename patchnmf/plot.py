import numpy as np
import matplotlib.pyplot as plt

def plot_nmf_t(nmf_t, gt_acts=None, plot_gt=False):

    fig, axs = plt.subplots(nmf_t.components_.shape[0], 1, figsize=(10, nmf_t.components_.shape[0]))

    for i in range(nmf_t.components_.shape[0]):
        axs[i].plot(nmf_t.components_[i, :]/np.max(nmf_t.components_[i, :]), c='C1')
        if plot_gt:
            axs[i].plot(gt_acts[i]/np.max(gt_acts[i]), c='C0')
            
        axs[i].axis('off')

    plt.show()

def plot_nmf_px(nmf_px, xy_px):
    
    n_components = nmf_px.components_.shape[0]
    #plotting nmf components
    fig, axs = plt.subplots(n_components, dpi=1000)

    rois_auto = []

    for i in range(0,n_components):
        loading = nmf_px.components_[i,:]
        loading_img = loading.reshape(xy_px, xy_px)
        axs[i].imshow(loading_img)
        axs[i].axis('off')