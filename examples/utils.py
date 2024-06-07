import math
import numpy as np
import torch


def rotate_joint(bd_pose, jidx):
    jidx_l = jidx*3
    print(f'Joint pose before {jidx+1} {bd_pose[:, jidx_l:jidx_l+3]}')
    bd_pose[:, jidx_l:jidx_l+3] = np.array([0, 1, 0]) * math.pi * 0.5
    print(f'Joint pose after {jidx+1} {bd_pose[:, jidx_l:jidx_l+3]}')
    return bd_pose

def extract_gt_full(rot_fp: str) -> tuple:
    rot_data = np.load(rot_fp, allow_pickle=True)
    transl = rot_data["transl"]
    global_orient = rot_data["global_orient"]
    expression = rot_data["expression"]
    body_pose = rot_data["body_pose"]
    betas = rot_data["betas"]
    lhpose = rot_data["left_hand_pose"]
    rhpose = rot_data["right_hand_pose"]

    if transl.shape[0] > 1:
        frame_id = 100
        transl = transl[frame_id][np.newaxis, ...]
        global_orient = global_orient[frame_id][np.newaxis, ...]
        expression = expression[frame_id][np.newaxis, ...]
        body_pose = body_pose[frame_id][np.newaxis, ...]
        betas = betas[frame_id][np.newaxis, ...]
        lhpose = lhpose[frame_id][np.newaxis, ...]
        rhpose = rhpose[frame_id][np.newaxis, ...]

    transl = torch.from_numpy(transl).type(torch.float32)
    global_orient = torch.from_numpy(global_orient).type(torch.float32)
    expression = torch.from_numpy(expression).type(torch.float32)
    body_pose = torch.from_numpy(body_pose).type(torch.float32)
    betas = torch.from_numpy(betas).type(torch.float32)
    left_hand_pose = torch.from_numpy(lhpose).type(torch.float32)
    right_hand_pose = torch.from_numpy(rhpose).type(torch.float32)

    print(f"{betas.shape=}")
    print(f"{global_orient=}")
    print(f"{transl=}")
    print(f"{expression.shape=}")
    print(f"{body_pose.shape=}")
    return transl, global_orient, expression, body_pose, betas, left_hand_pose, right_hand_pose


def extract_val(rot_fp: str, fr_id: int) -> tuple:
    pred = np.load(rot_fp, allow_pickle=True)["pred_rotations"]
    gt = np.load(rot_fp, allow_pickle=True)["gt_rotations"]
    t = torch.float32

    pred = pred[fr_id]
    gt = gt[fr_id]

    go_pred = pred[0, :]
    go_gt = gt[0, :]
    go_pred = torch.from_numpy(go_pred).type(t).unsqueeze(0)
    go_gt = torch.from_numpy(go_gt).type(t).unsqueeze(0)

    bp_pred = pred[1:22, :]
    bp_gt = gt[1:22, :]
    bp_pred = torch.from_numpy(bp_pred).type(t).unsqueeze(0)
    bp_gt = torch.from_numpy(bp_gt).type(t).unsqueeze(0)

    lh_pred = np.zeros((1, 15, 3))
    lh_gt = np.zeros((1, 15, 3))
    lh_pred[:, [2, 8], :] = pred[[22, 24], :]
    lh_gt[:, [2, 8], :] = gt[[22, 24], :]
    lh_pred = torch.from_numpy(lh_pred).type(t)
    lh_gt = torch.from_numpy(lh_gt).type(t)

    rh_pred = np.zeros((1, 15, 3))
    rh_gt = np.zeros((1, 15, 3))
    rh_pred[:, [2, 8], :] = pred[[22, 24], :]
    rh_gt[:, [2, 8], :] = gt[[23, 25], :]
    rh_pred = torch.from_numpy(rh_pred).type(t)
    rh_gt = torch.from_numpy(rh_gt).type(t)

    return (go_pred, go_gt), (bp_pred, bp_gt), (lh_pred, lh_gt), (rh_pred, rh_gt)