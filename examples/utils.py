import math
import numpy as np
import torch
from pytorch3d import transforms

from fit3d_keypoint_order import Fit3DOrder26P


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


def load_data(rot_fp: str, fr_id: int) -> tuple:
    pred = np.load(rot_fp, allow_pickle=True)["pred_rotations"]
    gt = np.load(rot_fp, allow_pickle=True)["gt_rotations"]
    betas = np.load(rot_fp, allow_pickle=True)["betas"]

    pred = pred[fr_id]
    gt = gt[fr_id]
    betas = betas[fr_id]
    return pred, gt, betas

def extract_val(pred, gt):
    t = torch.float32

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


def flip_pose(pose: np.ndarray) -> np.ndarray:
    angles = np.linalg.norm(pose, axis=-1, keepdims=True)
    epsilon = 1e-10
    angles_safe = np.where(angles == 0, epsilon, angles)

    axes = pose / angles_safe

    norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    unit_vectors = axes / norms

    flip_pose = unit_vectors * angles_safe

    # flip_pose[1:, 1] *= -1
    # flip_pose[1:, 2] *= -1

    # flip_pose = flip_pose[Fit3DOrder26P.flip_lr_indices()]
    return flip_pose

def flip_matrix(pose: np.ndarray) -> np.ndarray:
    rot_mat = transforms.axis_angle_to_matrix(pose)
    return transforms.matrix_to_axis_angle(rot_mat)


def magnitude(x: object) -> object:
    eps = 1e-15
    if isinstance(x, np.ndarray):
        x = np.power(x, 2)
        x = np.sum(x, axis=-1)
        return np.sqrt(x)[:, np.newaxis] + eps
    else:
        x = torch.pow(x, 2)
        x = torch.sum(x, dim=-1)
        nonzero = x != 0
        x[nonzero] = torch.sqrt(x[nonzero])
        return x.unsqueeze(-1)

def quaternion_angle(x: np.ndarray) -> np.ndarray:
    assert (magnitude(x) > 0.9999).all()
    return 2 * np.arccos(x[:, 0])[:, np.newaxis]

def calc_vector(x: np.ndarray) -> tuple:
    origins = np.zeros_like(x)
    for org, goal in Fit3DOrder26P._kinematic_tree:
        if goal >= x.shape[0] or org >= x.shape[0]:
            break
        origins[goal, :] = x[org, :]
    out = x - origins
    return out, origins

def form_unit_axis_angle(x: np.ndarray, angles: np.ndarray) -> np.ndarray:
    x = x / magnitude(x)
    assert (magnitude(x) > 0.9999).all()
    aa = angles * x
    return aa

def form_quaternion(vec: np.ndarray, angles: np.ndarray) -> np.ndarray:
    vec = vec / magnitude(vec)
    assert (magnitude(vec) > 0.9999).all()
    w = np.cos(angles / 2)
    x = np.sin(angles / 2) * vec[:, 0, np.newaxis]
    y = np.sin(angles / 2) * vec[:, 1, np.newaxis]
    z = np.sin(angles / 2) * vec[:, 2, np.newaxis]
    quat = np.concatenate([w, x, y, z], axis=1)
    return quat



def joints_to_axis_angle(
    angles: torch.Tensor,
    joints: torch.Tensor,
    betas: torch.Tensor,
    model
) -> torch.Tensor:
    angles = angles.clone()
    joints = joints.clone()
    assert angles.shape[0] == joints.shape[0] and joints.shape[0] == betas.shape[0]

    bs, _, _ = joints.shape
    zero_joints, _ = smplx_tpose_joints(betas, model)
    joints -= joints[:, 0].clone()

    P_old = collect_positions(zero_joints)
    P_new = collect_positions(joints)

    rot_axis = []
    for _, (po, pn) in enumerate(zip(P_old, P_new)):
        if po is None:
            axis = torch.zeros((bs, 3))
        else:
            if len(po.shape) == 3:
                po = torch.mean(po, dim=-1)
                pn = torch.mean(pn, dim=-1)
        # elif len(po.shape) == 3:
        #     R = find_rotation_matrix_svd_batch(po, pn)
        #     axis = form_axis_from_rot_mat(R)
        # else:
            axis = torch.cross(po, pn, dim=-1)
            axis = axis / torch.linalg.vector_norm(axis, ord=2, dim=-1)
        rot_axis.append(axis)

    rot_axis = torch.stack(rot_axis, dim=1)
    rot_axis = torch.nan_to_num(rot_axis)

    rot_axis[:, 1:22] *= angles
    return rot_axis


def find_rotation_matrix_svd_batch(p_old, p_new) -> torch.Tensor:
    H = p_new @ p_old.transpose(-2, -1)
    U, _, V = torch.svd(H)
    Vt = torch.transpose(V, -2, -1)
    R = torch.matmul(U, Vt)

    det_batch = torch.det(R)
    neg_det_mask = det_batch < 0
    U[neg_det_mask, :, -1] *= -1
    R = torch.matmul(U, Vt.transpose(-2, -1))
    return R


def form_axis_from_rot_mat(R: torch.Tensor) -> torch.Tensor:
    bs, _, _ = R.shape
    trace_R = torch.einsum("...ii", R)

    angles = torch.arccos((trace_R - 1) / 2).clamp(-1.0, 1.0)  # Ensure values are within valid range
    axes = torch.zeros(bs, 3, dtype=torch.float32)

    small_angle_mask = angles < 1e-8
    large_angle_mask = (angles - math.pi).abs() < 1e-8
    general_mask = ~(small_angle_mask | large_angle_mask)

    # Handle small angles (close to 0)
    axes[small_angle_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    # Handle large angles (close to 180 degrees)
    if large_angle_mask.any():
        R_plus_I = R[large_angle_mask] + torch.eye(3, dtype=torch.float32)
        axes[large_angle_mask] = torch.nn.functional.normalize(R_plus_I[:, :, 0], dim=1)

    # Handle general case
    if general_mask.any():
        R_gen = R[general_mask]
        rx = R_gen[:, 2, 1] - R_gen[:, 1, 2]
        ry = R_gen[:, 0, 2] - R_gen[:, 2, 0]
        rz = R_gen[:, 1, 0] - R_gen[:, 0, 1]
        axis_gen = torch.stack((rx, ry, rz), dim=1)
        axes[general_mask] = torch.nn.functional.normalize(axis_gen, dim=1)

    return axes


def smplx_tpose_joints(betas: torch.Tensor, model) -> torch.Tensor:
    bs, _ = betas.shape
    zero_smplx_params = {
        "body_pose": torch.zeros((bs, 21, 3)),
        "global_orient": torch.zeros((bs, 3)),
        "betas": betas,
        "return_verts": True,
        "plot_joints": True,
    }
    zero_model = model(**zero_smplx_params)
    zero_joints = zero_model.joints.detach().clone()
    zero_joints -= zero_joints[:, 0].clone()
    return zero_joints, zero_model


def collect_positions(joints: torch.Tensor) -> list:
    children = [[] for _ in range(Fit3DOrder26P._num_joints)]

    for parent, child in Fit3DOrder26P._kinematic_tree:
        if child > 21:
            break
        children[parent].append(joints[:, child].clone())

    out = []
    for vs in children:
        if len(vs) > 1:
            out.append(torch.stack(vs, -1))
        elif len(vs) == 1:
            out.append(vs[0])
        else:
            out.append(None)
    return out