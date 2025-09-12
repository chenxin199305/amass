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
comp_device = "cpu"  # Using GPU may cause unexpected issues for some users

# ====================================================================================================

amass_npz_fname = osp.join(support_dir, "github_data/dmpl_sample.npz")  # the path to body data
bdata = numpy.load(amass_npz_fname)

# you can set the gender manually and if it differs from data"s then contact or
# interpenetration issues might happen
subject_gender = (bdata["gender"].tolist()).decode()

print("Data keys available:%s" % list(bdata.keys()))

print("The subject of the mocap sequence is  {}.".format(subject_gender))

# ====================================================================================================

from human_body_prior.body_model.body_model import BodyModel

bm_fname = osp.join(support_dir, f"body_model/smplh/{subject_gender}/model.npz")
dmpl_fname = osp.join(support_dir, f"body_model/dmpls/{subject_gender}/model.npz")

num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
faces = c2c(bm.f)

# ====================================================================================================

time_length = len(bdata["trans"])

body_parms = {
    "root_orient": torch.Tensor(bdata["poses"][:, :3]).to(comp_device),  # controls the global root orientation
    "pose_body": torch.Tensor(bdata["poses"][:, 3:66]).to(comp_device),  # controls the body
    "pose_hand": torch.Tensor(bdata["poses"][:, 66:]).to(comp_device),  # controls the finger articulation
    "trans": torch.Tensor(bdata["trans"]).to(comp_device),  # controls the global body position
    "betas": torch.Tensor(numpy.repeat(bdata["betas"][:num_betas][numpy.newaxis], repeats=time_length, axis=0)).to(comp_device),  # controls the body shape. Body shape is static
    "dmpls": torch.Tensor(bdata["dmpls"][:, :num_dmpls]).to(comp_device)  # controls soft tissue dynamics
}

print("Body parameter vector shapes: \n{}".format(" \n".join(["{}: {}".format(k, v.shape) for k, v in body_parms.items()])))
print("time_length = {}".format(time_length))

# ====================================================================================================

import trimesh

from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image
from body_visualizer.tools.vis_tools import imagearray2file

imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

# ====================================================================================================

w_dmpl_parms = {k: v for k, v in body_parms.items() if k in ["pose_body", "betas", "pose_hand", "dmpls"]}
w_dmpl_trans = torch.zeros_like(body_parms["trans"])
w_dmpl_trans[:, 0] += -0.5
w_dmpl_parms["trans"] = w_dmpl_trans
body_dmpls = bm(**w_dmpl_parms)

wo_dmpl_parms = {k: v for k, v in body_parms.items() if k in ["pose_body", "betas", "pose_hand"]}
wo_dmpl_trans = torch.zeros_like(body_parms["trans"])
wo_dmpl_trans[:, 0] += 0.5
wo_dmpl_parms["trans"] = wo_dmpl_trans
body_wo_dmpls = bm(**wo_dmpl_parms)


def vis_dmpl_comparision():
    image_arr = []
    for fId in range(time_length):
        body_mesh_w_dmpl = trimesh.Trimesh(vertices=c2c(body_dmpls.v[fId]), faces=faces, vertex_colors=numpy.tile(colors["grey"], (6890, 1)))
        body_mesh_wo_dmpl = trimesh.Trimesh(vertices=c2c(body_wo_dmpls.v[fId]), faces=faces, vertex_colors=numpy.tile(colors["grey"], (6890, 1)))

        mv.set_static_meshes([body_mesh_w_dmpl, body_mesh_wo_dmpl])
        body_image = mv.render(render_wireframe=False)
        image_arr.append(body_image)

    image_arr = numpy.array(image_arr).reshape([1, 1, -1, 1600, 1600, 3])
    imagearray2file(image_arr, outpath=f"{support_dir}/dmpl_sample.gif", fps=60)
    imagearray2file(image_arr, outpath=f"{support_dir}/dmpl_sample.mp4", fps=60)


vis_dmpl_comparision()
