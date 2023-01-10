# barrel-patch_analysis
Code I contributed for Sofia's project containing NMF patch analysis, some simulations etc.

Example gif of a patch simulation used to benchmark patch-extraction techniques:

![](https://github.com/juremaj/barrel-patch_analysis/blob/main/docs/media/sim_anim.gif)

The code here also performs cross-validation on the NMF. Here's a result from the above simulation where the ground truth number of components is known:


<img src="https://github.com/juremaj/barrel-patch_analysis/blob/main/docs/media/cv_nmf.png" alt="drawing" width="300"/>


# Installation

First clone the repo and cd into the directory:

```
git clone https://github.com/juremaj/barrel-patch_analysis
cd barrel-patch_analysis
```

You can set up the environment simply by using the environment.yaml file:

`conda env create -f environment.yml`

This will ensure the same versions of dependencies are used. But it can also be a bit messy since there are a lot of JupyterLab dependencies installed + it might be platform dependent (the one here is tested only on Linux).

Alternatively we can also just set up a new environment, activate it:

```
conda create --name patchnmf
conda activate patchnmf
```

and install the latest versions of the dependencies by running:

```
conda install -c conda-forge jupyterlab
conda install -c conda-forge numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install -c conda-forge scikit-image
```

Finally we need to install the local `patchnmf` library by running:

```
pip install -e .
```


# Usage

First we need to add some data, within a `data` folder under `barrel-patch_analysis`. This repo assumes the same organization as `deve-networks` (see [readme](https://github.com/juremaj/deve-networks#organising-data)). Once we have some data installed we can simply lunch jupyter lab:

```
jupyter lab
```

And start doing some analysis :)

## Demos

This repo contains some useful functions for loading data from tiffs, preprocessing them, running NMf (with cross-validation), running simulations etc. There are two demo notebooks going through the main functionalities, these can be found in:

`notebooks_demo/`

It's best to leave these unchanged and use `notebooks_dev` to develop new code (for example there is a notebook there that can be used to compare the NMF outputs across datasets. You can then always refer to the use cases in `notebooks_demo` if something gets messed up.
