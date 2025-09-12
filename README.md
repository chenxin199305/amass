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

1. Install dependencies:

```shell
# Create conda environment
conda create -n amass python=3.7 -y
conda activate amass

# Install required packages
bash install_env.sh
```

2. Change dependency code to enable plot image using matplotlib (Due to codebase bugs):

```python
# In body_visualizer/tools/vis_tools.py
import numpy as np
import cv2
import os
import trimesh


def show_image(img_ndarray):
    '''
    Visualize rendered body images resulted from render_smpl_params in Jupyter notebook
    :param img_ndarray: Nxim_hxim_wx3
    '''
    import matplotlib.pyplot as plt
    import cv2
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')

    plt.show()  # ADD THIS LINE

    # fig.canvas.draw()
    # return True


def imagearray2file(img_array, outpath=None, fps=30):
    '''
    :param nparray: RxCxTxwidthxheightx3
    :param outpath: the directory where T images will be dumped for each time point in range T
    :param fps: fps of the gif file
    :return:
        it will return an image list with length T
        if outpath is given as a png file, an image will be saved for each t in T.
        if outpath is given as a gif file, an animated image with T frames will be created.
    '''
    # content ...

    # "fps" is deprecated argument, use "duration" instead in new version of imageio
    with imageio.get_writer(outpath, mode='I', duration=(1000 * 1 / fps)) as writer:
        pass

    # content ...
```

### Run Scripts

Run examples as following:

```bash
python pyscript/01-AMASS_Visualization.py
python pyscript/02-AMASS_DNN.py
python pyscript/03-AMASS_Visualization_Advanced.py
python pyscript/04-AMASS_DMPL.py
```