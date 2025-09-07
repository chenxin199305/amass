# AMASS: Archive of Motion Capture as Surface Shapes

![alt text](support_data/github_data/datasets_preview.png "Samples of bodies in AMASS recovered from Motion Capture sequences")

[AMASS](http://amass.is.tue.mpg.de) is a large database of human motion unifying different optical marker-based motion capture datasets by representing them within a common framework and parameterization.
AMASS is readily useful for animation, visualization, and generating training data for deep learning.

Here we provide tools and tutorials to use AMASS in your research projects. More specifically:

- Following the recommended splits of data by AMASS, we provide three non-overlapping train/validation/test splits.
- AMASS uses an extended version of [SMPL+H](http://mano.is.tue.mpg.de/) with [DMPLs](https://smpl.is.tue.mpg.de/).
  Here we show how to load different components and visualize a body model with AMASS data.
- AMASS is also compatible with [SMPL](http://smpl.is.tue.mpg.de) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models.
  We show how to use the body data from AMASS to animate these models.

## Body Models

AMASS uses [MoSh++](https://amass.is.tue.mpg.de) pipeline to fit [SMPL+H body model](https://mano.is.tue.mpg.de/)
to human optical marker based motion capture (mocap) data.
In the paper we use SMPL+H with extended shape space, i.e. 16 betas, and 8 [DMPLs](https://smpl.is.tue.mpg.de/).
Please download models and place them them in body_models folder of this repository after you obtained the code from GitHub.

### Environment Setup

```shell
# Create conda environment
conda create -n amass python=3.7 -y
conda activate amass

# Install required packages
bash install_env.sh
```
