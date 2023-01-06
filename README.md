# barrel-patch_analysis
Code I contributed for Sofia's project containing NMF patch analysis, some simulations etc.

![](https://github.com/juremaj/barrel-patch_analysis/blob/main/docs/media/sim_anim.gif)

(Example gif of a patch simulation used to benchmark patch-extraction techniques)

NOTE (5th January 2023): this repo currently ISN'T runnable, since it assumes dependencies on the devenw library (not included within). I will refactor this code in the following weeks so it will be easy to use for everyone.

I have transferred (probably most of) the parts of the devenw library that are unique to this project in the 'library' folder, since these are not needed for devenw within the scope of my msc. There might still be some functions in teh devenw library that are needed by both though (think a bit about what to do in this case).

This also performs cross-validation on the NMF. Here's a result from the above simulation where the ground truth number of components is known:


<img src="https://github.com/juremaj/barrel-patch_analysis/blob/main/docs/media/cv_nmf.png" alt="drawing" width="300"/>


# Installation

You can set up the environment simply by using the environment.yaml file (see conda documentation), which will ensure the same versions are used. Alternatively we can also just install the latest versions of the dependencies by running:

```
conda install -c conda-forge jupyterlab
conda install -c conda-forge numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
```
