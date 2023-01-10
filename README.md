# barrel-patch_analysis
Code I contributed for Sofia's project containing NMF patch analysis, some simulations etc.

Example gif of a patch simulation used to benchmark patch-extraction techniques:

![](https://github.com/juremaj/barrel-patch_analysis/blob/main/docs/media/sim_anim.gif)

The code here also performs cross-validation on the NMF. Here's a result from the above simulation where the ground truth number of components is known:


<img src="https://github.com/juremaj/barrel-patch_analysis/blob/main/docs/media/cv_nmf.png" alt="drawing" width="300"/>


# Installation

You can set up the environment simply by using the environment.yaml file:

`conda env create -f environment.yml`

This will ensure the same versions of dependencies are used. But it can also be a bit messy since there are a lot of JupyterLab dependencies installed + it might be platform dependent (the one here is tested only on Linux).

Alternatively we can also just install the latest versions of the dependencies by running:

```
conda install -c conda-forge jupyterlab
conda install -c conda-forge numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install -c conda-forge scikit-image
```
