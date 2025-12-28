import os
import numpy as np
from scipy.io import loadmat

# Make paths relative to this script so the converter can be run from any CWD
BASE_DIR = os.path.join(os.path.dirname(__file__), "3DMM")

def _np(path):
    return os.path.join(BASE_DIR, path)

mat_path = _np("01_MorphableModel.mat")
if not os.path.exists(mat_path):
    raise FileNotFoundError(f"BFM .mat not found at {mat_path}. Place 01_MorphableModel.mat in {BASE_DIR}")

original_BFM = loadmat(mat_path)

topo_path = _np("topology_info.npy")
if not os.path.exists(topo_path):
    raise FileNotFoundError(f"topology_info.npy not found at {topo_path}. Run the placeholder generator or add the official file.")

sub_inds = np.load(topo_path, allow_pickle=True).item()["sub_inds"]

shapePC = original_BFM["shapePC"]
shapeEV = original_BFM["shapeEV"]
shapeMU = original_BFM["shapeMU"]
texPC = original_BFM["texPC"]
texEV = original_BFM["texEV"]
texMU = original_BFM["texMU"]

b_shape = shapePC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)
mu_shape = shapeMU.reshape(-1, 3)

b_tex = texPC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)
mu_tex = texMU.reshape(-1, 3)

b_shape = b_shape[:, sub_inds, :].reshape(199, -1)
mu_shape = mu_shape[sub_inds, :].reshape(-1)
b_tex = b_tex[:, sub_inds, :].reshape(199, -1)
mu_tex = mu_tex[sub_inds, :].reshape(-1)

exp_info = np.load(_np("exp_info.npy"), allow_pickle=True).item()

out_path = _np("3DMM_info.npy")
np.save(
    out_path,
    {
        "mu_shape": mu_shape,
        "b_shape": b_shape,
        "sig_shape": shapeEV.reshape(-1),
        "mu_exp": exp_info["mu_exp"],
        "b_exp": exp_info["base_exp"],
        "sig_exp": exp_info["sig_exp"],
        "mu_tex": mu_tex,
        "b_tex": b_tex,
        "sig_tex": texEV.reshape(-1),
    },
)
print(f"Wrote 3DMM info to {out_path}")
