import os
import numpy as np
from tqdm import tqdm
import torch
from pytorch3d import transforms

import sys
sys.path.insert(0, '/home/mmc-user/3d-hpe-rot/3d_hpe_code/dataset')

from fit3d.utils import read_meta, get_pathes, read_data

DATASET_ROOT = "/data/datasets/fit3d"
SUBSET = "train"
META_PATH = os.path.join(DATASET_ROOT, "fit3d_info.json")
CAM_NAMES, SUBJECTS, ACTIONS = read_meta(META_PATH)

if __name__ == "__main__":
    for sbj in tqdm(SUBJECTS):
        sbj_acts = ACTIONS[sbj]
        for act in sbj_acts:
            for cam in CAM_NAMES[:1]:
                outdir = os.path.join(DATASET_ROOT, SUBSET, sbj, "smplx_npz")
                os.makedirs(outdir, exist_ok=True)
                out_fp = os.path.join(outdir, f"{act}.npz")

                pathes = get_pathes(DATASET_ROOT, SUBSET, sbj, act, cam)
                _, _, _, smplx_params = read_data(*pathes)
                smplz_params_new = {}
                for k, v in smplx_params.items():
                    if k in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose"]:
                        v = torch.Tensor(v)
                        aa = transforms.matrix_to_axis_angle(v)
                        smplz_params_new[k] = aa.detach().numpy()
                    else:
                        smplz_params_new[k] = v
                np.savez(out_fp, **smplz_params_new)