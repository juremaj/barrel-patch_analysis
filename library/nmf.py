import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA, NMF
from matplotlib import animation
from scipy.ndimage import gaussian_filter, convolve1d
from scipy.signal import convolve2d
import os

# for cvNMF
from numpy.random import randn, rand
from scipy.optimize import minimize
from devenw.nnls import nnlsm_blockpivot as nnlstsq
import itertools
from scipy.spatial.distance import cdist


def downsample_tiff_avg(tiff, n=4):

    tiff_ds = []
    for i in range(tiff.shape[0]):
        kernel = np.ones((n, n))
        convolved = convolve2d(tiff[i,:,:], kernel, mode='valid')
        this_tiff_ds = convolved[::n, ::n] / n
        #tiff_ds = np.concatenate((tiff_ds, this_tiff_ds))
        tiff_ds.append(this_tiff_ds)

    plt.imshow(tiff[i,:,:])
    plt.title('original frame example')
    plt.show()
    plt.imshow(this_tiff_ds)
    plt.title('downsampled frame example')
    plt.show()
    tiff = np.array(tiff_ds)

    
    return tiff


def poiss_train(tau, t_max=10):
    # tau - parameter of exponential distribution
    # t_max - duration of spike train
    
    st = []
    st.append(np.random.exponential(tau)) # first spike time
    count = 0
    
    while st[count] < t_max:
        st.append(st[count] + np.random.exponential(tau)) # subsequent spike times with ISIs from exponential distribution
        count += 1
    
    # removing final spike (outside [0, 10])
    st = st[0:-1]
    
    return np.array(st), count

# from Williams cvNMF implementation (http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/)

def censored_lstsq(A, B, M):
    """Solves least squares problem with missing data in B
    Note: uses a broadcasted solve for speed.
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    if A.ndim == 1:
        A = A[:,None]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    try:
        # transpose to get r x n
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T
    except:
        r = T.shape[1]
        T[:,np.arange(r),np.arange(r)] += 1e-6
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T

def censored_nnlstsq(A, B, M):
    """Solves nonnegative least-squares problem with missing data in B
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)
    Returns
    -------
    X (ndarray) : nonnegative r x n matrix that minimizes norm(M*(AX - B))
    """
    if A.ndim == 1:
        A = A[:,None]
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    X = np.empty((B.shape[1], A.shape[1]))
    for n in range(B.shape[1]):
        X[n] = nnlstsq(T[n], rhs[n], is_input_prod=True)[0].T
    return X.T

def cv_pca(data, rank, M=None, p_holdout=0.3, nonneg=False):
    """Fit PCA or NMF while holding out a fraction of the dataset.
    """

    # choose solver for alternating minimization
    if nonneg:
        solver = censored_nnlstsq
    else:
        solver = censored_lstsq

    # create masking matrix
    if M is None:
        M = np.random.rand(*data.shape) > p_holdout

    # initialize U randomly
    if nonneg:
        U = np.random.rand(data.shape[0], rank)
    else:
        U = np.random.randn(data.shape[0], rank)

    # fit pca/nmf
    for itr in range(50):
        Vt = solver(U, data, M)
        U = solver(Vt.T, data.T, M.T).T

    # return result and test/train error
    resid = np.dot(U, Vt) - data
    train_err = np.mean(resid[M]**2)
    test_err = np.mean(resid[~M]**2)
    return U, Vt, train_err, test_err

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
