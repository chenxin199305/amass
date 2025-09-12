# Description: DNN training on AMASS dataset using SMPLH body model

import numpy
import torch

# ====================================================================================================

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp

# from ipywidgets import interact_manual
# from ipywidgets import IntSlider

current_file_path = osp.abspath(__file__)
current_dir = osp.dirname(current_file_path)
project_dir = osp.dirname(current_dir)

support_dir = osp.join(project_dir, "support_data")

# Choose the device to run the body model on.
# comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_device = "cpu"  # Using GPU may cause unexpected issues for some users

# ====================================================================================================

amass_npz_fname = osp.join(support_dir, 'github_data/amass_sample.npz')  # the path to body data
bdata = numpy.load(amass_npz_fname)

num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']

print('Data keys available:%s' % list(bdata.keys()))

print('The subject of the mocap sequence is  {}.'.format(subject_gender))

# ====================================================================================================

time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(compute_device),  # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(compute_device),  # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(compute_device),  # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(compute_device),  # controls the global body position
    'betas': torch.Tensor(numpy.repeat(bdata['betas'][:num_betas][numpy.newaxis], repeats=time_length, axis=0)).to(compute_device),  # controls the body shape. Body shape is static
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(compute_device)  # controls soft tissue dynamics
}

print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k, v.shape) for k, v in body_parms.items()])))
print('time_length = {}'.format(time_length))

# ====================================================================================================

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import show_image

imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

## SMPL-X

# ====================================================================================================

from human_body_prior.body_model.body_model import BodyModel

bm_smpl_fname = osp.join(support_dir, 'body_model/smplh/{}/model.npz'.format(subject_gender))
# bm_smpl_fname = '/is/ps3/nghorbani/code-repos/amass/support_data/body_model/smpl/neutral/model.npz'

bm = BodyModel(bm_fname=bm_smpl_fname, num_betas=num_betas).to(compute_device)

faces = c2c(bm.f)
num_verts = bm.init_v_template.shape[1]

# ====================================================================================================

print({k: v.shape for k, v in body_parms.items() if k in ['pose_body', 'betas']})
body = bm(**{k: v.to(compute_device) for k, v in body_parms.items() if k in ['pose_body']})
body_mesh_wofingers = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=numpy.tile(colors['grey'], (num_verts, 1)))
mv.set_static_meshes([body_mesh_wofingers])
body_image_wofingers = mv.render(render_wireframe=False)
show_image(body_image_wofingers)
