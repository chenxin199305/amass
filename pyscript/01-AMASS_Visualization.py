# Description: Visualize AMASS mocap data using SMPLH and DMPL models from human_body_prior

import numpy
import torch

# ====================================================================================================

from os import path as osp
from human_body_prior.tools.omni_tools import copy2cpu as c2c

current_file_path = osp.abspath(__file__)
current_dir = osp.dirname(current_file_path)
project_dir = osp.dirname(current_dir)

support_dir = osp.join(project_dir, "support_data")

print(
    f"Current file path: {current_file_path}\n"
    f"Current directory: {current_dir}\n"
    f"Project directory: {project_dir}\n"
    f"Support directory: {support_dir}\n"
)

# Choose the device to run the body model on.
# comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
comp_device = "cpu"  # Using GPU may cause unexpected issues for some users

amass_npz_fname = osp.join(support_dir, "github_data/dmpl_sample.npz")  # the path to body data
bdata = numpy.load(amass_npz_fname)

# you can set the gender manually and if it differs from data"s then contact or interpenetration issues might happen
subject_gender = bdata["gender"]
subject_gender_str = subject_gender.item().decode('utf-8')

print(
    f"Data keys available: {list(bdata.keys())}\n"
    f"The subject gender of the mocap sequence is {subject_gender_str} {type(subject_gender_str)}\n"
)

# ====================================================================================================

from human_body_prior.body_model.body_model import BodyModel

bm_fname = osp.join(support_dir, f"body_model/smplh/{subject_gender_str}/model.npz")
dmpl_fname = osp.join(support_dir, f"body_model/dmpls/{subject_gender_str}/model.npz")

num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
faces = c2c(bm.f)

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

print(
    f"\n",
    f"=" * 50 + "\n",
    f"Visualize betas and pose_body\n",
    f"=" * 50 + "\n",
)

import trimesh

from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

body_pose_beta = bm(**{k: v for k, v in body_parms.items() if k in [
    "pose_body",
    "betas",
]})


def vis_body_pose_beta(fId=0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=numpy.tile(colors["grey"], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


print(
    f"Body vertices (v) shape: {body_pose_beta.v.shape}\n"
    f"Body vertices (v) min: {body_pose_beta.v.min()}\n"
    f"Body vertices (v) max: {body_pose_beta.v.max()}\n"
    f"Body vertices (v) mean: {body_pose_beta.v.mean()}\n"
    f"Body vertices (v) std: {body_pose_beta.v.std()}\n"
    f"Body faces shape: {faces.shape}\n"
    f"Body faces min: {faces.min()}\n"
    f"Body faces max: {faces.max()}\n"
    f"Body faces mean: {faces.mean()}\n"
    f"Body faces std: {faces.std()}\n"
)

vis_body_pose_beta(fId=0)

# ====================================================================================================

print(
    f"\n",
    f"=" * 50 + "\n",
    f"Visualize pose hands\n",
    f"=" * 50 + "\n",
)

body_pose_hand = bm(**{k: v for k, v in body_parms.items() if k in [
    "pose_body",
    "betas",
    "pose_hand",
]})


def vis_body_pose_hand(fId=0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=numpy.tile(colors["grey"], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


vis_body_pose_hand(fId=0)

# ====================================================================================================

print(
    f"\n",
    f"=" * 50 + "\n",
    f"Visualize body joints\n",
    f"=" * 50 + "\n",
)


def vis_body_joints(fId=0):
    joints = c2c(body_pose_hand.Jtr[fId])
    joints_mesh = points_to_spheres(joints, point_color=colors["red"], radius=0.005)

    mv.set_static_meshes([joints_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


vis_body_joints(fId=0)

# ====================================================================================================

print(
    f"\n",
    f"=" * 50 + "\n",
    f"Visualize DMPLs\n",
    f"=" * 50 + "\n",
)

body_dmpls = bm(**{k: v for k, v in body_parms.items() if k in [
    "pose_body",
    "betas",
    "pose_hand",
    "dmpls",
]})


def vis_body_dmpls(fId=0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_dmpls.v[fId]), faces=faces, vertex_colors=numpy.tile(colors["grey"], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


vis_body_dmpls(fId=0)

# ====================================================================================================

print(
    f"\n",
    f"=" * 50 + "\n",
    f"Visualize global root orientation and translation\n",
    f"=" * 50 + "\n",
)

"""
Jason 2025-09-11:
可以理解为，AMASS 数据集提供了人体模型在不同时间点的
- 全身姿态（pose_body）
- 手部姿态（pose_hand）
- 形状参数（betas）
- 软组织动态参数（dmpls）
- 全局位置（trans）
- 全局方向（root_orient）
这些参数共同描述了人体在三维空间中的完整状态。
"""
body_trans_root = bm(**{k: v for k, v in body_parms.items() if k in [
    "pose_body",
    "betas",
    "pose_hand",
    "dmpls",
    "trans",
    "root_orient",
]})


def vis_body_trans_root(fId=0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=numpy.tile(colors["grey"], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


vis_body_trans_root(fId=0)


def vis_body_transformed(fId=0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=numpy.tile(colors["grey"], (6890, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (1, 0, 0)))

    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


vis_body_transformed(fId=0)
