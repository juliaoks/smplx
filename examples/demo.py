# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import argparse
import pickle

import numpy as np
import torch

import smplx

import math

from pytorch3d import transforms

# from utils import extract_gt_full, rotate_joint, extract_val, load_data, flip_pose
from utils import *

def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=True,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False,
         extract="gt",
         rot_fp=None,
         flip_gt=False,
         ):

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext, use_pca=False)

    if extract == "gt":
        # bd_pose = rotate_joint(bd_pose, 16)

        (transl, global_orient, expression, body_pose,
            betas, left_hand_pose, right_hand_pose) = extract_gt_full(rot_fp)

        output = model(
            betas=betas,
            expression=expression,
            return_verts=True,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            plot_joints=False,
            left_hand_pose=left_hand_pose,
            use_pca=False,
            right_hand_pose=right_hand_pose,
        )
        output = [output]
        joints = output.joints.detach().cpu().numpy().squeeze()
        print("lhip, lknee, lankle ->\n", joints[[1, 4, 7]], "\n") # lhip, lknee, lankle
        print("left_hip, pelvis, right_hip ->\n", joints[[1, 0, 2]], "\n")
        print("left_shoulder, left_elbow, left_wrist ->\n", joints[[16, 18, 20]], "\n")
    elif extract == "val" and flip_gt:
        fr_id = 0
        print(f"Frame Id={fr_id}")
        _, gt = load_data(rot_fp, fr_id)

        if flip_gt:
            flip_gt = flip_pose(gt)
        (go_pred, go_gt), (bp_pred, bp_gt), (lh_pred, lh_gt), (rh_pred, rh_gt) = extract_val(flip_gt, gt)

        global_orient, body_pose = go_gt, bp_gt
        left_hand_pose, right_hand_pose = lh_gt, rh_gt

        betas, expression = None, None
        output_gt = model(
            betas=betas,
            expression=expression,
            return_verts=True,
            global_orient=global_orient,
            body_pose=body_pose,
            plot_joints=False,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )

        global_orient, body_pose = go_pred, bp_pred
        left_hand_pose, right_hand_pose = lh_pred, rh_pred

        output_pred = model(
            betas=betas,
            expression=expression,
            return_verts=True,
            global_orient=global_orient,
            body_pose=body_pose,
            plot_joints=False,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        output = [output_gt, output_pred]
    elif extract == "val":
        # fr_id = 43789
        fr_id = 37620
        # fr_id = 110
        print(f"Frame Id={fr_id}")
        # fr_id = 160
        pred, gt, betas = load_data(rot_fp, fr_id)
        betas = torch.from_numpy(betas).unsqueeze(0)

        if flip_gt:
            gt = flip_pose(gt)

        (go_pred, go_gt), (bp_pred, bp_gt), (lh_pred, lh_gt), (rh_pred, rh_gt) = extract_val(pred, gt)

        global_orient, body_pose = go_gt, bp_gt
        left_hand_pose, right_hand_pose = lh_gt, rh_gt

        expression = None
        output_gt = model(
            betas=betas,
            expression=expression,
            return_verts=True,
            global_orient=global_orient,
            body_pose=body_pose,
            plot_joints=False,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )

        global_orient, body_pose = go_pred, bp_pred
        left_hand_pose, right_hand_pose = lh_pred, rh_pred

        output_pred = model(
            betas=betas,
            expression=expression,
            return_verts=True,
            global_orient=global_orient,
            body_pose=body_pose,
            plot_joints=False,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        output = [output_gt, output_pred]
    else:
        betas, expression = None, None
        betas = torch.zeros((1, model.num_betas))
        expression = torch.zeros((1, model.num_expression_coeffs))
        # if sample_shape:
        #     betas = torch.randn([1, model.num_betas], dtype=torch.float32)
        # if sample_expression:
        #     expression = torch.randn(
        #         [1, model.num_expression_coeffs], dtype=torch.float32)

        deg = 20
        rad = np.deg2rad(deg)

        global_orient = torch.zeros((1, 3))
        body_pose = torch.zeros((1, 21, 3))
        transl = torch.zeros((1, 3))
        output1 = model(
            betas=betas.clone(),
            expression=expression.clone(),
            body_pose=body_pose.clone(),
            global_orient=global_orient.clone(),
            transl=transl.clone(),
            return_verts=True,
            plot_joints=True,
        )

        def reconstruct_body_pose(joints_new, joints_old, angles) -> torch.Tensor:
            bs = joints_old.shape[0]
            assert joints_old.shape == joints_new.shape

            pelvis_old = joints_old[:, 0].clone()
            pelvis_new = joints_new[:, 0].clone()
            joints_old -= pelvis_old
            joints_new -= pelvis_new

            children_new = [[] for _ in range(Fit3DOrder26P._num_joints)]
            children_old = [[] for _ in range(Fit3DOrder26P._num_joints)]
            for parent, child in Fit3DOrder26P._kinematic_tree:
                if child > 21:
                    break
                children_new[parent].append(joints_new[:, child].clone())
                children_old[parent].append(joints_old[:, child].clone())

            def stack_vec(vectors):
                out = []
                for vs in vectors:
                    if len(vs) > 1:
                        out.append(torch.stack(vs, -1))
                    elif len(vs) == 1:
                        out.append(vs[0])
                    else:
                        out.append(None)
                return out

            P_old = stack_vec(children_old)
            P_new = stack_vec(children_new)
            P_old_t = [torch.transpose(p, dim0=-2, dim1=-1) if isinstance(p, torch.Tensor) else p for p in P_old]

            rot_axis = []
            for idx, (po, po_t, pn) in enumerate(zip(P_old, P_old_t, P_new)):
                if po is None:
                    no_axis = torch.zeros((bs, 3))
                    rot_axis.append(no_axis)
                elif len(po.shape) == 3:
                    H = pn @ po_t
                    u, s, v = torch.svd(H)
                    vt = torch.transpose(v, -2, -1)
                    r = torch.matmul(u, vt)
                    trace_r = torch.einsum("...ii", r)
                    theta = torch.acos((trace_r - 1) / 2)
                    u = torch.zeros(bs, 3)
                    u[:, 0] = r[:, 2, 1] - r[:, 1, 2]
                    u[:, 1] = r[:, 0, 2] - r[:, 2, 0]
                    u[:, 2] = r[:, 1, 0] - r[:, 0, 1]
                    u *= 1 / (2 * torch.sin(theta))
                    rot_axis.append(u)
                else:
                    u = torch.cross(po, pn, dim=-1)
                    u = u / torch.linalg.vector_norm(u, ord=2, dim=-1)
                    rot_axis.append(u)

            rot_axis = torch.stack(rot_axis, axis=1)
            rot_axis = torch.nan_to_num(rot_axis)
            rot_axis[:, 1:22] *= angles
            return rot_axis

        # transl = torch.tensor([[0, 1, 0]])
        # global_orient = torch.Tensor([[0, 0, 1]])
        # global_orient = global_orient / magnitude(global_orient)
        # global_orient *= rad

        test_list = [
            # Fit3DOrder26P.left_ankle,
            # Fit3DOrder26P.right_ankle,
            # Fit3DOrder26P.left_knee,
            # Fit3DOrder26P.right_knee,
            # Fit3DOrder26P.left_elbow,
            # Fit3DOrder26P.right_elbow,
            # Fit3DOrder26P.spine3,
            # Fit3DOrder26P.spine2,
            Fit3DOrder26P.spine1,
        ]
        axis = torch.Tensor([1, 1, 1])
        for idx in test_list:
            body_pose[:, idx-1] = axis
        mag = torch.linalg.vector_norm(body_pose, ord=2, dim=-1).unsqueeze(-1)
        mag_safe = torch.where(mag == 0, torch.tensor(1e-10), mag.clone())
        body_pose = body_pose / mag_safe
        body_pose *= rad

        output2 = model(
            betas=betas.clone(),
            expression=expression.clone(),
            body_pose=body_pose.clone(),
            global_orient=global_orient.clone(),
            transl=transl.clone(),
            return_verts=True,
            plot_joints=True,
        )
        print(f"{output2.body_pose=}")
        # print("body joints", output2.joints[:, :22] == output1.joints[:, :22])
        # print("hand joints", output2.joints[:, 26:55] == output1.joints[:, 26:55])

        # print("right_hand_pose", output2.right_hand_pose == output1.right_hand_pose)
        # print("left_hand_pose", output2.left_hand_pose == output1.left_hand_pose)

        print(f"{output2.joints[:, :22] - output1.joints[:, :22]}")

        joints = output2.joints[:, :22].detach().clone()
        # joints -= output1.joints[:, :22].detach().clone()
        pelvis = joints[:, 0].clone()
        joints -= pelvis
        # print(f"output2 pelvis rel {joints=}")
        # for parent, kid in Fit3DOrder26P._kinematic_tree:
        #     if kid > 21:
        #         continue
        #     joints[:, kid] = joints[:, kid] - joints[:, parent]

        # # angles = torch.zeros((1, 21, 1))
        angles = magnitude(output2.body_pose.detach().clone())

        body_pose_new_rec = reconstruct_body_pose(
            output2.joints[:, :22].detach().clone(),
            output1.joints[:, :22].detach().clone(),
            angles,
        )
        body_pose_2 = body_pose_new_rec[:, 1:22].clone()

        # vec = torch.zeros((1, 21, 3))
        joints_old = (output1.joints[:, 1:22] - output1.joints[:, 0]).detach().clone()
        # joints_old = output1.joints[:, :22].detach().clone()
        # for parent, kid in Fit3DOrder26P._kinematic_tree:
        #     if kid > 21:
        #         continue
        #     joints_old[:, kid] = joints_old[:, kid] - joints_old[:, parent]
        # joints_old = joints_old[:, 1:].clone()

        vec = joints[:, 1:].clone()
        vec = torch.cross(joints_old, vec, dim=-1)
        vec2 = torch.zeros((1, 21, 3))
        # for idx, parent in test_list:
        #     vec[:, idx] = joints[:, idx] - joints[:, parent]
        nkids = torch.zeros((1, 21, 1))
        for parent, kid in Fit3DOrder26P._kinematic_tree:
            if kid > 21 or parent == 0:
                continue
            # if parent == 9 and kid in [14, 13]:
            #     continue
            # print(f"{parent=} {kid=}")
            # vec[:, parent-1] -= joints[:, kid-1]
            vec2[:, parent-1] += vec[:, kid-1]
            nkids[:, parent-1] += 1

        nkids = torch.where(nkids == 0, 1, nkids)
        vec2 = vec2 / nkids

        vec = vec2.clone()

        # vec = torch.cross(joints_old, vec, dim=-1)

        mag = magnitude(vec)
        mag_safe = torch.where(mag == 0, torch.tensor(1e-10), mag)
        vec = vec / mag_safe
        # vec *= angles

        K = torch.Tensor(
            [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
        ).unsqueeze(0).unsqueeze(0).repeat(1, vec.shape[1], 1, 1)
        K[:, :, 2, 1] *= vec[:, :, 0]
        K[:, :, 1, 2] *= vec[:, :, 0]
        K[:, :, 2, 0] *= vec[:, :, 1]
        K[:, :, 0, 2] *= vec[:, :, 1]
        K[:, :, 1, 0] *= vec[:, :, 2]
        K[:, :, 0, 1] *= vec[:, :, 2]

        I = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, vec.shape[1], 1, 1)
        angles_sin = torch.sin(angles.unsqueeze(-1))
        angles_cos = 1 - torch.cos(angles.unsqueeze(-1))
        R = I + angles_sin * K + angles_cos * (K @ K)

        vec = transforms.matrix_to_axis_angle(R)
        vec = vec.type(torch.float32)
        # print(f"{vec=}")
        # # vec, origins = calc_vector(output2.joints[:, :22].squeeze().detach().numpy())
        # # vec[2, [0, 1]] *= 0
        # # vec[2, [2]] *= -1
        # # unit_aa = form_unit_axis_angle(vec, angles)[np.newaxis, 1:]

        output3 = model(betas=betas.clone(),
                        expression=expression.clone(),
                        global_orient=global_orient.clone(),
                        body_pose=vec.clone(),
                        return_verts=True, plot_joints=True)
        print(f"{output3.body_pose=}")
        output4 = model(betas=betas.clone(),
                        expression=expression.clone(),
                        global_orient=global_orient.clone(),
                        body_pose=body_pose_2.clone(),
                        return_verts=True, plot_joints=True)
        print(f"{output4.body_pose=}")
        # print(f"{output2.joints[:, :22] - output1.joints[:, :22]}")
        # print(f"{output3.joints[:, :22] - output1.joints[:, :22]}")
        # print(f"{output4.joints[:, :22] - output1.joints[:, :22]}")
        # print("body joints", output2.joints[:, :22] == output3.joints[:, :22])
        # print("body pose", output2.body_pose == output3.body_pose)
        # output = [output1, output2]
        # output = [output1, output2, output3]
        output = [output2, output3, output4]


    if plotting_module == 'pyrender':
        import pyrender
        import trimesh

        tx, ty, tz = 0, -0.5, 1.8
        rot_rad = np.deg2rad(0)
        translation_matrix = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        rot_matrix = np.array([
            [np.cos(rot_rad), 0, np.sin(rot_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(rot_rad), 0, np.cos(rot_rad), 0],
            [0, 0, 0, 1]
        ])
        # cam_pose = np.array([[1, 0, 0, 0],
        #                         [0, 1, 0, -.5],
        #                         [0, 0, 1, 1.8],
        #                         [0, 0, 0, 1]])
        cam_pose = np.dot(translation_matrix, rot_matrix)

        # prepare camera and light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        camera = pyrender.OrthographicCamera(xmag=0.05, ymag=0.5)

        scene = pyrender.Scene(
            # bg_color=[0.6, 0.6, 0.6, 0.1],
            # ambient_light=(0.3, 0.3, 0.3)
        )
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)

        name = ["Ground Truth", "Predict", "Predict Reconstruct"]
        colors = [[50, 245, 39, 0.5], [245, 39, 243, 0.5], [255,215,0, 0.5]]
        for idx, out in enumerate(output):
            if not out:
                continue
            if len(output) > 1:
                print(f"{idx=} {name[idx]}")

            # idx = idx + 1
            vertices = out.vertices.detach().cpu().numpy().squeeze()
            joints = out.joints.detach().cpu().numpy().squeeze()

            # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3 * idx, 0.3, 0.3, 0.5]
            vertex_colors = np.array(colors[idx])
            vertex_colors[:3] = vertex_colors[:3] / 255
            tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                    vertex_colors=vertex_colors,
                                    process=False)

            mesh = pyrender.Mesh.from_trimesh(tri_mesh,)
            scene.add(mesh, 'mesh')

            if plot_joints:
                sm = trimesh.creation.uv_sphere(radius=0.009)
                vertex_colors[-1] = 1.0
                sm.visual.vertex_colors = vertex_colors
                tfs = np.tile(np.eye(4), (Fit3DOrder26P._num_joints, 1, 1))
                tfs[:22, :3, 3] = joints[Fit3DOrder26P._body_smplx]
                tfs[22:, :3, 3] = joints[Fit3DOrder26P._fingers_smplx]
                joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True, viewer_flags=dict(show_mesh_axes=True))
    elif plotting_module == 'matplotlib':
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
        plt.show()
    elif plotting_module == 'open3d':
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices)
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry = [mesh]
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            joints_pcl.points = o3d.utility.Vector3dVector(joints)
            joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
            geometry.append(joints_pcl)

        o3d.visualization.draw_geometries(geometry)
    else:
        raise ValueError('Unknown plotting_module: {}'.format(plotting_module))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    # parser.add_argument('--model-folder', required=True, type=str,
    #                     help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    parser.add_argument("--extract", type=str,)
    parser.add_argument("--rot-fp", type=str,)
    parser.add_argument("--flip-gt", action='store_true')

    args = parser.parse_args()

    model_folder = "/Users/yuliiaoks/Documents/uni_aux/master/models/"

    # model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = True
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression
    extract = args.extract
    rot_fp = args.rot_fp
    flip_gt = args.flip_gt

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour,
         extract=extract,
         rot_fp=rot_fp,
         flip_gt=flip_gt,
         )
