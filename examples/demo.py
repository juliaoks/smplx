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

from utils import extract_gt_full, rotate_joint, extract_val

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

    elif extract == "val":
        # fr_id = 43789
        fr_id = 160
        (go_pred, go_gt), (bp_pred, bp_gt), (lh_pred, lh_gt), (rh_pred, rh_gt) = extract_val(rot_fp, fr_id)

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
    else:
        betas, expression = None, None
        if sample_shape:
            betas = torch.randn([1, model.num_betas], dtype=torch.float32)
        if sample_expression:
            expression = torch.randn(
                [1, model.num_expression_coeffs], dtype=torch.float32)

        output = model(betas=betas, expression=expression,
                   return_verts=True, plot_joints=True)
        output = [output]

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

        name = ["Ground Truth", "Predict"]
        colors = [[50, 245, 39, 0.5], [245, 39, 243, 0.5]]
        for idx, out in enumerate(output):
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
                tfs = np.tile(np.eye(4), (26, 1, 1))
                tfs[:22, :3, 3] = joints[:22]
                tfs[22:, :3, 3] = joints[[27, 42, 33, 48]]
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
         )
