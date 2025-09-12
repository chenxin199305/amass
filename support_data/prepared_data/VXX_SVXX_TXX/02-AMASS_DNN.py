# Description: DNN training on AMASS dataset using SMPLH body model

import numpy
import torch

# ====================================================================================================

import os
from os import path as osp

current_file_path = osp.abspath(__file__)
current_dir = osp.dirname(current_file_path)
project_dir = osp.dirname(current_dir)

support_dir = osp.join(project_dir, "support_data")

# Choose the device to run the body model on.
# comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
comp_device = "cpu"  # Using GPU may cause unexpected issues for some users

# ====================================================================================================

from human_body_prior.tools.omni_tools import log2file, makepath
from human_body_prior.tools.omni_tools import copy2cpu as c2c

expr_code = "VXX_SVXX_TXX"  # VERSION_SUBVERSION_TRY

msg = """ Initial use of standard AMASS dataset preparation pipeline """

amass_dir = f"{support_dir}/amass_npz"  # "PATH_TO_DOWNLOADED_NPZFILES/*/*_poses.npz"
work_dir = f"{support_dir}/prepared_data/VXX_SVXX_TXX"

logger = log2file(makepath(work_dir, "%s.log" % (expr_code), isfile=True))
logger("[%s] AMASS Data Preparation Began." % expr_code)
logger(msg)

# amass_splits = {
#     "vald": ["HumanEva", "MPI_HDM05", "SFU", "MPI_mosh"],
#     "test": ["Transitions_mocap", "SSM_synced"],
#     "train": ["CMU", "MPI_Limits", "TotalCapture", "Eyes_Japan_Dataset", "KIT",
#               "BML", "EKUT", "TCD_handMocap", "ACCAD"]
# }
amass_splits = {
    "vald": ["SFU", ],
    "test": ["SSM_synced"],
    "train": ["MPI_Limits"]
}
amass_splits["train"] = list(set(amass_splits["train"]).difference(set(amass_splits["test"] + amass_splits["vald"])))

# ====================================================================================================

from amass.data.prepare_data import prepare_amass

prepare_amass(amass_splits, amass_dir, work_dir, logger=logger)

# ====================================================================================================

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob


class AMASS_DS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, num_betas=16):
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, "*.pt")):
            k = os.path.basename(data_fname).replace(".pt", "")
            self.ds[k] = torch.load(data_fname)
        self.num_betas = num_betas

    def __len__(self):
        return len(self.ds["trans"])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        data["root_orient"] = data["pose"][:3]
        data["pose_body"] = data["pose"][3:66]
        data["pose_hand"] = data["pose"][66:]
        data["betas"] = data["betas"][:self.num_betas]

        return data


num_betas = 16  # number of body parameters
testsplit_dir = os.path.join(work_dir, "stage_III", "test")

ds = AMASS_DS(dataset_dir=testsplit_dir, num_betas=num_betas)
print("Test split has %d datapoints." % len(ds))

batch_size = 5
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

# ====================================================================================================

import trimesh
from body_visualizer.tools.vis_tools import colors, imagearray2file
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import show_image

imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

# ====================================================================================================

from human_body_prior.body_model.body_model import BodyModel

bm_fname = osp.join(support_dir, "body_models/smplh/male/model.npz")

num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas).to(comp_device)
faces = c2c(bm.f)

bdata = next(iter(dataloader))
body_v = bm.forward(**{k: v.to(comp_device) for k, v in bdata.items() if k in ["pose_body", "betas"]}).v

view_angles = [0, 180, 90, -90]
images = numpy.zeros([len(view_angles), batch_size, 1, imw, imh, 3])
for cId in range(0, batch_size):

    orig_body_mesh = trimesh.Trimesh(vertices=c2c(body_v[cId]), faces=c2c(bm.f), vertex_colors=numpy.tile(colors["grey"], (6890, 1)))

    for rId, angle in enumerate(view_angles):
        if angle != 0: orig_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(numpy.radians(angle), (0, 1, 0)))
        mv.set_meshes([orig_body_mesh], group_name="static")
        images[rId, cId, 0] = mv.render()

        if angle != 0: orig_body_mesh.apply_transform(trimesh.transformations.rotation_matrix(numpy.radians(-angle), (0, 1, 0)))

img = imagearray2file(images)
show_image(numpy.array(img)[0])
