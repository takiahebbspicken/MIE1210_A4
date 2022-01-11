import time
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from pathlib import Path
from scipy import linalg, sparse
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, gmres



class Boundary:
    def __init__(self, boundary_type, boundary_value, boundary_location):
        self.create_boundary(boundary_type, boundary_value, boundary_location)

    def create_boundary(self, boundary_type, boundary_value, boundary_location):
        self.type = boundary_type
        self.value = boundary_value
        self.location = boundary_location


class Mesh:
    def __init__(self, mesh_type, boundaries_u, boundaries_v, boundaries_p, x_points, y_points, length_x, length_y):
        self.type = mesh_type
        self.rhos = 1
        self.mus = 1
        self.create_mesh(boundaries_u, boundaries_v, boundaries_p)
        self.is_ref_node = False
        self.is_object = False
        if mesh_type.lower() == '2D_uniform'.lower():
            self.generate_2d_uniform_mesh(x_points, y_points, length_x, length_y)
            self.cell_volumes_areas_deltas()

    def create_mesh(self, boundaries_u, boundaries_v, boundaries_p):
        for boundary_u, boundary_v, boundary_p in zip(boundaries_u, boundaries_v, boundaries_p):
            if boundary_u.location.lower() == 'left':
                self.u_left_boundary = boundary_u
            elif boundary_u.location.lower() == 'right':
                self.u_right_boundary = boundary_u
            elif boundary_u.location.lower() == 'top':
                self.u_top_boundary = boundary_u
            elif boundary_u.location.lower() == 'bottom':
                self.u_bottom_boundary = boundary_u
            if boundary_v.location.lower() == 'left':
                self.v_left_boundary = boundary_v
            elif boundary_v.location.lower() == 'right':
                self.v_right_boundary = boundary_v
            elif boundary_v.location.lower() == 'top':
                self.v_top_boundary = boundary_v
            elif boundary_v.location.lower() == 'bottom':
                self.v_bottom_boundary = boundary_v
            if boundary_p.location.lower() == 'left':
                self.p_left_boundary = boundary_p
            elif boundary_p.location.lower() == 'right':
                self.p_right_boundary = boundary_p
            elif boundary_p.location.lower() == 'top':
                self.p_top_boundary = boundary_p
            elif boundary_p.location.lower() == 'bottom':
                self.p_bottom_boundary = boundary_p

    def generate_2d_uniform_mesh(self, x_points, y_points, length_x, length_y, r=1):
        if r == 1:
            d_x = length_x / x_points
            deltas_x = np.ones((x_points + 1, 1)) * d_x
            deltas_x[0] = d_x / 2
            deltas_x[-1] = d_x / 2
            d_y = length_y / y_points
            deltas_y = np.ones((y_points + 1, 1)) * d_y
            deltas_y[0] = d_y / 2
            deltas_y[-1] = d_y / 2
        # Deltas when there is inflation
        elif r > 1:
            deltas_x = np.zeros((x_points + 1, 1))
            deltas_y = np.zeros((y_points + 1, 1))
            deltas_x[0] = ((1 - r) / (1 - r ** (x_points / 2))) * (length_x / 2)
            for k in range(1, x_points + 1):
                coord = np.sum(deltas_x)
                if coord <= (length_x / 2):
                    deltas_x[k] = r * deltas_x[k - 1]
                else:
                    deltas_x[k] = deltas_x[k - 1] / r
            deltas_y[0] = ((1 - r) / (1 - r ** (y_points / 2))) * (length_y / 2)
            for k in range(1, y_points + 1):
                coord = np.sum(deltas_y)
                if coord <= (length_y / 2):
                    deltas_y[k] = r * deltas_y[k - 1]
                else:
                    deltas_y[k] = deltas_y[k - 1] / r
        # Scale deltas to ensure mesh has not been extended past the actual domain
        if np.sum(deltas_x) > length_x:
            rescale_factor = length_x / np.sum(deltas_x)
            deltas_x = deltas_x * rescale_factor
        if np.sum(deltas_y) > length_y:
            rescale_factor = length_y / np.sum(deltas_y)
            deltas_y = deltas_y * rescale_factor
        self.xgrid = x_points
        self.ygrid = y_points
        self.lx = length_x
        self.ly = length_y
        self.dx = deltas_x
        self.dy = deltas_y
        self.num_nodes = (len(self.dx) - 1) * (len(self.dy) - 1)
        self.x_coordinates = np.cumsum(self.dx[0:-1])
        self.y_coordiantes = np.cumsum(self.dy[0:-1])
        self.column = [int(np.floor(i / self.ygrid)) for i in range(self.num_nodes)]

        self.vel = np.zeros((self.num_nodes, 2))  # [u, v]
        self.u_vel_boundaries = np.zeros((self.num_nodes, 4))  # [E N W S] boundary velocity conditions
        self.v_vel_boundaries = np.zeros((self.num_nodes, 4))  # [E N W S]
        self.vel_correction = np.zeros((self.num_nodes, 2))  # [u, v]
        self.vel_correction_boundaries = np.zeros((self.ygrid + 2, 4))

        self.pressure = np.zeros((self.num_nodes, 1))
        self.pressure_boundaries = np.zeros((self.ygrid, 4))  # [LEFT TOP RIGHT BOTTOM] boundary values
        self.pressure_correction = np.zeros((self.num_nodes, 1))
        self.pressure_correction_boundaries = np.zeros((self.ygrid, 4))  # [LEFT TOP RIGHT BOTTOM] boundary values

        self.vel_face = np.zeros((self.num_nodes, 4))  # [E N W S]
        self.vel_face_boundaries = np.zeros((self.num_nodes, 4))
        self.vel_face_correction = np.zeros((self.num_nodes, 4))  # [E N W S]
        self.vel_face_correction_boundaries = np.zeros((self.num_nodes, 4))

        self.a_momentum = np.zeros((self.num_nodes, 5))  # [P E N W S]
        self.momentum_source_u = np.zeros((self.num_nodes, 2))  # [Su_u Su_v]
        self.momentum_source_pp = np.zeros((self.num_nodes, 2))  # [Spp_u Spp_v]
        self.momentum_pressure_source = np.zeros((self.num_nodes, 2))  # [S_pp_X, S_pp_Y]
        self.u_boundary_idx = np.ones((self.num_nodes, 4))  # 1 if node is not next to wall boundary, else 0 [E N W S]
        self.v_boundary_idx = np.ones((self.num_nodes, 4))

        self.a_pressure = np.zeros((self.num_nodes, 5))  # [P E N W S]
        self.pressure_source = np.zeros((self.num_nodes, 1))
        self.a_pressure_boundary = np.ones((self.num_nodes, 4)
                                           )  # 1 if node is not next to D u/v boundary, else 0 [E N W S]

    def cell_volumes_areas_deltas(self):
        self.volumes = np.zeros((self.num_nodes, 1))
        self.areas = np.zeros((self.num_nodes, 2))  # 0 column is Ay, 1 column is Ax
        for i in range(self.num_nodes):
            if i == 0:  # Bottom left boundary corner
                self.areas[i, 0] = self.dy[0] + self.dy[1] / 2  # Ay
                self.areas[i, 1] = self.dx[0] + self.dx[1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i == self.ygrid - 1:  # Top left corner boundary
                self.areas[i, 0] = self.dy[-1] + self.dy[-2] / 2  # Ay
                self.areas[i, 1] = self.dx[0] + self.dx[1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i == self.num_nodes - self.ygrid:  # Bottom right corner boundary
                self.areas[i, 0] = self.dy[0] + self.dy[1] / 2  # Ay
                self.areas[i, 1] = self.dx[-1] + self.dx[-2] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i == self.num_nodes - 1:  # Top right corner boundary
                self.areas[i, 0] = self.dy[-1] + self.dy[-2] / 2  # Ay
                self.areas[i, 1] = self.dx[-1] + self.dx[-2] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i < self.ygrid:  # Left boundary
                self.areas[i, 0] = self.dy[i + 1] / 2 + self.dy[i] / 2  # Ay
                self.areas[i, 1] = self.dx[0] + self.dx[1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i >= self.num_nodes - self.ygrid:  # Right boundary
                self.areas[i, 0] = self.dy[(i - self.column[i] * self.ygrid) + 1] / 2 + \
                                   self.dy[i - self.column[i] * self.ygrid] / 2  # Ay
                self.areas[i, 1] = self.dx[-1] + self.dx[-2] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i % self.ygrid == 0 and i != 0:  # Bottom boundary
                self.areas[i, 0] = self.dy[0] + self.dy[1] / 2  # Ay
                self.areas[i, 1] = self.dx[self.column[i]] / 2 + self.dx[self.column[i] + 1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i % self.ygrid == (self.ygrid - 1):  # Top boundary
                self.areas[i, 0] = self.dy[-1] + self.dy[-2] / 2  # Ay
                self.areas[i, 1] = self.dx[self.column[i]] / 2 + self.dx[self.column[i] + 1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            else:
                self.areas[i, 0] = self.dy[(i - self.column[i] * self.ygrid) + 1] / 2 + \
                                   self.dy[i - self.column[i] * self.ygrid] / 2  # Ay
                self.areas[i, 1] = self.dx[self.column[i]] / 2 + self.dx[self.column[i] + 1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]

    def create_object(self, size, position):
        self.is_object = True
        self.obj_idxs = np.ones((self.num_nodes, 1))
        self.obj_p = position
        self.obj_s = size
        x_length = size[0]
        y_length = size[1]
        start_x = position[0]
        start_y = position[1]
        max_col = int(np.ceil(self.xgrid * x_length))
        max_row = int(np.ceil(self.xgrid * y_length))
        for j in range(max_col):
            self.obj_idxs[start_y + j * self.ygrid: max_row + j * self.ygrid] = 0
            self.momentum_source_pp[start_y + j * self.ygrid: max_row + j * self.ygrid, 0] = -10 ** 40
            self.momentum_source_pp[start_y + j * self.ygrid: max_row + j * self.ygrid, 1] = -10 ** 40
            self.momentum_source_u[start_y + j * self.ygrid: max_row + j * self.ygrid, 0] = 0
            self.momentum_source_u[start_y + j * self.ygrid: max_row + j * self.ygrid, 1] = 0

    def set_re(self, re):
        self.re = re

    def set_reference_node(self, ref_node):
        self.ref_node = (ref_node[0] - 1) + self.ygrid * (ref_node[1] - 1)
        self.ref_node_value = ref_node[2]
        self.ref_pressure_source = np.zeros((self.num_nodes, 1))
        self.is_ref_node = True

    def set_rho(self, rhos):
        self.rhos = rhos

    def set_mus(self, mus):
        self.mus = mus

    def set_gamma(self, gammas):
        self.gammas = gammas


# Functions
def set_boundary_values(mesh):
    # U boundaries
    if mesh.u_left_boundary.type == 'D':
        mesh.vel_face[0:mesh.ygrid, 2] = mesh.u_left_boundary.value
        mesh.u_vel_boundaries[0:mesh.ygrid, 2] = mesh.u_left_boundary.value
        mesh.a_pressure_boundary[0:mesh.ygrid, 2] = 0
    if mesh.u_right_boundary.type == 'D':
        mesh.vel_face[mesh.num_nodes - mesh.ygrid:, 0] = mesh.u_right_boundary.value
        mesh.u_vel_boundaries[mesh.num_nodes - mesh.ygrid:, 0] = mesh.u_right_boundary.value
        mesh.a_pressure_boundary[mesh.num_nodes - mesh.ygrid:, 0] = 0
    elif mesh.u_right_boundary.type is None:
        pass
    if mesh.u_bottom_boundary.type == 'D':
        mesh.u_vel_boundaries[0::mesh.ygrid, 3] = mesh.u_bottom_boundary.value
    if mesh.u_top_boundary.type == 'D':
        mesh.u_vel_boundaries[mesh.ygrid - 1::mesh.ygrid, 1] = mesh.u_top_boundary.value

    # V boundaries
    if mesh.v_left_boundary.type == 'D':
        mesh.v_vel_boundaries[0:mesh.ygrid, 2] = mesh.v_left_boundary.value
    if mesh.v_right_boundary.type == 'D':
        mesh.v_vel_boundaries[mesh.num_nodes - mesh.ygrid:, 0] = mesh.v_right_boundary.value
    if mesh.v_bottom_boundary.type == 'D':
        mesh.vel_face[0::mesh.ygrid, 3] = mesh.v_bottom_boundary.value
        mesh.v_vel_boundaries[0::mesh.ygrid, 3] = mesh.v_bottom_boundary.value
        mesh.a_pressure_boundary[0::mesh.ygrid, 3] = 0
    if mesh.v_top_boundary.type == 'D':
        mesh.vel_face[mesh.ygrid - 1::mesh.ygrid, 1] = mesh.v_top_boundary.value
        mesh.v_vel_boundaries[mesh.ygrid - 1::mesh.ygrid, 1] = mesh.v_top_boundary.value
        mesh.a_pressure_boundary[mesh.ygrid - 1::mesh.ygrid, 1] = 0

    # Pressure conditions
    if mesh.p_left_boundary.type == 'N':
        mesh.pressure_boundaries[:, 0:1] = mesh.pressure[0:mesh.ygrid] - \
                                           mesh.p_left_boundary.value * mesh.dx[0]
        mesh.pressure_correction_boundaries[:, 0:1] = (mesh.pressure_correction[0:mesh.ygrid] -
                                                       mesh.p_left_boundary.value * mesh.dx[0]).reshape(mesh.ygrid, 1)
    if mesh.p_right_boundary.type == 'N':
        mesh.pressure_boundaries[:, 2:3] = mesh.pressure[mesh.num_nodes - mesh.ygrid:] - \
                                           mesh.p_right_boundary.value * mesh.dx[-1]
        mesh.pressure_correction_boundaries[:, 2:3] = (mesh.pressure_correction[mesh.num_nodes - mesh.ygrid:] -
                                                       mesh.p_right_boundary.value * mesh.dx[-1]).reshape(mesh.ygrid, 1)
    elif mesh.p_right_boundary.type == 'D':
        pass
    if mesh.p_bottom_boundary.type == 'N':
        mesh.pressure_boundaries[:, 3:4] = mesh.pressure[0::mesh.ygrid] - \
                                           mesh.p_bottom_boundary.value * mesh.dy[0]
        mesh.pressure_correction_boundaries[:, 3:4] = (mesh.pressure_correction[0::mesh.ygrid] -
                                                       mesh.p_bottom_boundary.value * mesh.dy[0]).reshape(mesh.ygrid, 1)
    if mesh.p_top_boundary.type == 'N':
        mesh.pressure_boundaries[:, 1:2] = mesh.pressure[mesh.ygrid - 1::mesh.ygrid] - \
                                           mesh.p_top_boundary.value * mesh.dy[-1]
        mesh.pressure_correction_boundaries[:, 1:2] = (mesh.pressure_correction[mesh.ygrid - 1::mesh.ygrid] -
                                                       mesh.p_top_boundary.value * mesh.dy[-1]).reshape(mesh.ygrid, 1)
    # Wall conditions
    if mesh.u_left_boundary.type == 'D':
        if mesh.v_left_boundary.type == 'D':
            mesh.u_boundary_idx[0:mesh.ygrid, 2] = 0
    if mesh.u_right_boundary.type == 'D':
        if mesh.v_right_boundary.type == 'D':
            mesh.u_boundary_idx[mesh.num_nodes - mesh.ygrid:, 0] = 0
    if mesh.u_top_boundary.type == 'D':
        if mesh.v_top_boundary.type == 'D':
            mesh.u_boundary_idx[mesh.ygrid - 1::mesh.ygrid, 1] = 0
    if mesh.u_bottom_boundary.type == 'D':
        if mesh.v_bottom_boundary.type == 'D':
            mesh.u_boundary_idx[0::mesh.ygrid, 3] = 0


def momentum_formulation(mesh):
    for idx in range(mesh.num_nodes):
        _, dpx_p, _, _, dpy_p, _ = pressure_derivatives(mesh, idx)
        f_e = mesh.rhos * mesh.vel_face[idx, 0] * mesh.areas[idx, 0]
        f_n = mesh.rhos * mesh.vel_face[idx, 1] * mesh.areas[idx, 1]
        f_w = mesh.rhos * mesh.vel_face[idx, 2] * mesh.areas[idx, 0]
        f_s = mesh.rhos * mesh.vel_face[idx, 3] * mesh.areas[idx, 1]
        d_e = mesh.areas[idx, 0] / mesh.dx[mesh.column[idx] + 1] / mesh.re
        d_n = mesh.areas[idx, 1] / mesh.dy[(idx % mesh.ygrid) + 1] / mesh.re
        d_w = mesh.areas[idx, 0] / mesh.dx[mesh.column[idx]] / mesh.re
        d_s = mesh.areas[idx, 1] / mesh.dy[(idx % mesh.ygrid)] / mesh.re
        # mesh.a_momentum[idx, 1] = (np.abs(f_e) - f_e) / 2 + d_e
        # mesh.a_momentum[idx, 2] = (np.abs(f_n) - f_n) / 2 + d_n
        # mesh.a_momentum[idx, 3] = (np.abs(f_w) + f_w) / 2 + d_w
        # mesh.a_momentum[idx, 4] = (np.abs(f_s) + f_s) / 2 + d_s
        mesh.a_momentum[idx, 1] = np.max([-f_e, 0]) + d_e
        mesh.a_momentum[idx, 2] = np.max([-f_n, 0]) + d_n
        mesh.a_momentum[idx, 3] = np.max([f_w, 0]) + d_w
        mesh.a_momentum[idx, 4] = np.max([f_s, 0]) + d_s

        # Central Differencing
        # mesh.a_momentum[idx, 1] = d_e - f_e / 2
        # mesh.a_momentum[idx, 2] = d_n - f_n / 2
        # mesh.a_momentum[idx, 3] = f_w / 2 + d_w
        # mesh.a_momentum[idx, 4] = f_s / 2 + d_s

        mesh.momentum_source_u[idx, 0] = mesh.u_vel_boundaries[idx, 0] * mesh.a_momentum[idx, 1] + \
                                         mesh.u_vel_boundaries[idx, 1] * mesh.a_momentum[idx, 2] + \
                                         mesh.u_vel_boundaries[idx, 2] * mesh.a_momentum[idx, 3] + \
                                         mesh.u_vel_boundaries[idx, 3] * mesh.a_momentum[idx, 4]
        mesh.momentum_source_u[idx, 1] = mesh.v_vel_boundaries[idx, 0] * mesh.a_momentum[idx, 1] + \
                                         mesh.v_vel_boundaries[idx, 1] * mesh.a_momentum[idx, 2] + \
                                         mesh.v_vel_boundaries[idx, 2] * mesh.a_momentum[idx, 3] + \
                                         mesh.v_vel_boundaries[idx, 3] * mesh.a_momentum[idx, 4]
        mesh.a_momentum[idx, 0] = mesh.a_momentum[idx, 1] + mesh.a_momentum[idx, 2] + \
                                  + mesh.a_momentum[idx, 3] + mesh.a_momentum[idx, 4] + \
                                  mesh.momentum_source_pp[idx, 1]

        # Stop contribution to a matrix if next to boundaries
        mesh.a_momentum[idx, 1] = mesh.a_momentum[idx, 1] * mesh.u_boundary_idx[idx, 0]
        mesh.a_momentum[idx, 2] = mesh.a_momentum[idx, 2] * mesh.u_boundary_idx[idx, 1]
        mesh.a_momentum[idx, 3] = mesh.a_momentum[idx, 3] * mesh.u_boundary_idx[idx, 2]
        mesh.a_momentum[idx, 4] = mesh.a_momentum[idx, 4] * mesh.u_boundary_idx[idx, 3]

        mesh.momentum_pressure_source[idx, 0] = -dpx_p * mesh.volumes[idx]
        mesh.momentum_pressure_source[idx, 1] = -dpy_p * mesh.volumes[idx]


def momentum_solver(mesh, uv_relax):
    # Format diagonals for A matrix
    upper_off_diag = -mesh.a_momentum[0:mesh.num_nodes - mesh.ygrid, 1]
    lower_off_diag = -mesh.a_momentum[mesh.ygrid:, 3]
    diag = mesh.a_momentum[:, 0]
    super_diag = -mesh.a_momentum[0:-1, 2]
    sub_diag = -mesh.a_momentum[1:, 4]
    diagonal = [diag, super_diag, sub_diag, upper_off_diag, lower_off_diag]
    diagonals_u = [diag / uv_relax, super_diag, sub_diag, upper_off_diag, lower_off_diag]
    diagonals_v = [diag / uv_relax, super_diag, sub_diag, upper_off_diag, lower_off_diag]

    # Compute b matrix
    b_matrix_u = mesh.momentum_pressure_source[:, 0] + mesh.momentum_source_u[:, 0] + ((1 - uv_relax) * diag /
                                                                                       uv_relax) * mesh.vel[:, 0]

    b_matrix_v = mesh.momentum_pressure_source[:, 1] + mesh.momentum_source_u[:, 1] + ((1 - uv_relax) * diag /
                                                                                       uv_relax) * mesh.vel[:, 1]

    # Form sparse matrix for solver
    a_sparse_matrix_u = diags(diagonals_u, [0, 1, -1, mesh.ygrid, -mesh.ygrid], format='csc')
    a_sparse_matrix_v = diags(diagonals_v, [0, 1, -1, mesh.ygrid, -mesh.ygrid], format='csc')
    u_solved = spla.splu(a_sparse_matrix_u).solve(b_matrix_u)
    v_solved = spla.splu(a_sparse_matrix_v).solve(b_matrix_v)

    # Check residuals of un-underrelaxed system
    a_mat = diags(diagonal, [0, 1, -1, mesh.ygrid, -mesh.ygrid], format='csc')
    b_mat_u = mesh.momentum_pressure_source[:, 0] + mesh.momentum_source_u[:, 0]
    b_mat_v = mesh.momentum_pressure_source[:, 1] + mesh.momentum_source_u[:, 1]
    residual_u = b_mat_u - a_mat * mesh.vel[:, 0]
    residual_v = b_mat_v - a_mat * mesh.vel[:, 1]

    # Update values for velocity
    mesh.vel[:, 0] = u_solved
    mesh.vel[:, 1] = v_solved

    # Return residuals
    return residual_u, residual_v


def face_velocities(mesh, uv_relax):
    for idx in range(mesh.num_nodes):
        # Compute pressure derivatives
        dpx_E, dpx_p, dpx_e, dpy_N, dpy_p, dpy_n = pressure_derivatives(mesh, idx)
        if idx == mesh.num_nodes - 1:  # Top right corner
            continue
        elif idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary adjacent
            mesh.vel_face[idx, 1] = ((mesh.vel[idx, 1] + mesh.vel[idx + 1, 1]) / 2
                                     + (1 / 2) * (mesh.volumes[idx] / mesh.a_momentum[idx, 0] * dpy_p
                                                  + mesh.volumes[idx + 1] /
                                                  mesh.a_momentum[idx + 1, 0] * dpy_N) -
                                     (mesh.volumes[idx] / 2 + mesh.volumes[idx + 1] / 2) *
                                     (1 / mesh.a_momentum[idx + 1, 0] + 1 / mesh.a_momentum[
                                         idx, 0]) * dpy_n) \
                                     * uv_relax + (1 - uv_relax) * mesh.vel_face[idx, 1]
        elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary adjacent
            mesh.vel_face[idx, 0] = ((mesh.vel[idx, 0] + mesh.vel[idx + mesh.ygrid, 0]) / 2
                                     + (1 / 2) * (mesh.volumes[idx] / mesh.a_momentum[idx, 0] * dpx_p
                                                  + mesh.volumes[idx + mesh.ygrid] /
                                                  mesh.a_momentum[idx + mesh.ygrid, 0] * dpx_E) -
                                     (mesh.volumes[idx] / 2 + mesh.volumes[idx + mesh.ygrid] / 2) *
                                     (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                                         idx, 0]) * dpx_e) \
                                     * uv_relax + (1 - uv_relax) * mesh.vel_face[idx, 0]
        else:
            mesh.vel_face[idx, 0] = ((mesh.vel[idx, 0] + mesh.vel[idx + mesh.ygrid, 0]) / 2
                                     + (1 / 2) * (mesh.volumes[idx] / mesh.a_momentum[idx, 0] * dpx_p
                                                  + mesh.volumes[idx + mesh.ygrid] /
                                                  mesh.a_momentum[idx + mesh.ygrid, 0] * dpx_E) -
                                     (mesh.volumes[idx] / 2 + mesh.volumes[idx + mesh.ygrid] / 2) *
                                     (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                                         idx, 0]) * dpx_e) \
                                     * uv_relax + (1 - uv_relax) * mesh.vel_face[idx, 0]
            mesh.vel_face[idx, 1] = ((mesh.vel[idx, 1] + mesh.vel[idx + 1, 1]) / 2
                                     + (1 / 2) * (mesh.volumes[idx] / mesh.a_momentum[idx, 0] * dpy_p
                                                  + mesh.volumes[idx + 1] /
                                                  mesh.a_momentum[idx + 1, 0] * dpy_N) -
                                     (mesh.volumes[idx] / 2 + mesh.volumes[idx + 1] / 2) *
                                     (1 / mesh.a_momentum[idx + 1, 0] + 1 / mesh.a_momentum[
                                         idx, 0]) * dpy_n) \
                                     * uv_relax + (1 - uv_relax) * mesh.vel_face[idx, 1]
        # Assign w and s face velocities
        if mesh.num_nodes - mesh.ygrid > idx:
            mesh.vel_face[idx + mesh.ygrid, 2] = mesh.vel_face[idx, 0]
        if idx % mesh.ygrid != (mesh.ygrid - 1):
            mesh.vel_face[idx + 1, 3] = mesh.vel_face[idx, 1]


def pressure_derivatives(mesh, idx):
    if idx < mesh.ygrid:  # Left boundary
        dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) \
                / (2 * mesh.areas[idx + mesh.ygrid, 1])
        dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2
                 * mesh.pressure_boundaries[idx, 0]) \
                / (2 * mesh.areas[idx, 1])
        dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
    elif idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary
        dpx_E = None
        dpx_p = (2 * mesh.pressure_boundaries[idx % mesh.ygrid, 2] -
                 mesh.pressure[idx - mesh.ygrid] -
                 mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
        dpx_e = None
    elif idx >= mesh.num_nodes - 2 * mesh.ygrid and idx < mesh.num_nodes - mesh.ygrid:  # Two from right boundary
        dpx_E = (2 * mesh.pressure_boundaries[idx % mesh.ygrid, 2] -
                 mesh.pressure[idx] - mesh.pressure[idx + mesh.ygrid]) \
                / (2 * mesh.areas[idx + mesh.ygrid, 1])
        dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
        dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
    else:
        dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 1])
        dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
        dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
    if idx % mesh.ygrid == 0:  # Bottom boundary
        dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + 1, 0])
        dpy_p = (mesh.pressure[idx + 1] + mesh.pressure[idx] -
                 2 * mesh.pressure_boundaries[mesh.column[idx], 3]) / (2 * mesh.areas[idx, 0])
        dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                2 * mesh.dy[idx % mesh.ygrid + 1])
    elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary
        dpy_N = None
        dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx]
                 - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        dpy_n = None
    elif idx % mesh.ygrid == (mesh.ygrid - 2):  # Two from top boundary
        dpy_N = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx] - mesh.pressure[
            idx + 1]) / (2 * mesh.areas[idx + 1, 0])
        dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (2 * mesh.dy[idx % mesh.ygrid + 1])
    else:
        dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + 1, 0])
        dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (2 * mesh.dy[idx % mesh.ygrid + 1])
    return dpx_E, dpx_p, dpx_e, dpy_N, dpy_p, dpy_n


def pressure_correction_formulation(mesh, uv_relax):
    for idx in range(mesh.num_nodes):
        if mesh.a_pressure_boundary[idx, 0] == 1:  # East node is not momentum dirichlet boundary
            mesh.a_pressure[idx, 1] = (mesh.rhos * mesh.areas[idx, 0] ** 2) * (
                    1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[idx, 0]) * uv_relax
        else:  # East node is a momentum dirichlet boundary
            mesh.a_pressure[idx, 1] = 0
        if mesh.a_pressure_boundary[idx, 1] == 1:
            mesh.a_pressure[idx, 2] = (mesh.rhos * mesh.areas[idx, 1] ** 2) * \
                                      (1 / mesh.a_momentum[idx + 1, 0] + 1 / mesh.a_momentum[idx, 0]) * uv_relax
        else:
            mesh.a_pressure[idx, 2] = 0
        if mesh.a_pressure_boundary[idx, 2] == 1:
            mesh.a_pressure[idx, 3] = (mesh.rhos * mesh.areas[idx, 0] ** 2) * (
                    1 / mesh.a_momentum[idx - mesh.ygrid, 0] + 1 / mesh.a_momentum[idx, 0]) * uv_relax
        else:
            mesh.a_pressure[idx, 3] = 0
        if mesh.a_pressure_boundary[idx, 3] == 1:
            mesh.a_pressure[idx, 4] = (mesh.rhos * mesh.areas[idx, 1] ** 2) * \
                                      (1 / mesh.a_momentum[idx - 1, 0] + 1 / mesh.a_momentum[idx, 0]) * uv_relax
        else:
            mesh.a_pressure[idx, 4] = 0
        mesh.a_pressure[idx, 0] = mesh.a_pressure[idx, 1] + mesh.a_pressure[idx, 2] + mesh.a_pressure[idx, 3] + \
                                  mesh.a_pressure[idx, 4]

        # Compute advection terms for mass imbalance (pressure correction source term)
        f_e = mesh.rhos * mesh.vel_face[idx, 0] * mesh.areas[idx, 0]
        f_n = mesh.rhos * mesh.vel_face[idx, 1] * mesh.areas[idx, 1]
        f_w = mesh.rhos * mesh.vel_face[idx, 2] * mesh.areas[idx, 0]
        f_s = mesh.rhos * mesh.vel_face[idx, 3] * mesh.areas[idx, 1]
        mesh.pressure_source[idx] = 2 * (f_w - f_e + f_s - f_n)

        # If reference node defined check for dirichlet conditions
        if mesh.is_ref_node:
            if idx == mesh.ref_node - 1:  # Node below reference node (north node is reference)
                mesh.ref_pressure_source[idx] = mesh.a_pressure[idx, 2] * mesh.ref_node_value
                mesh.a_pressure[idx, 2] = 0
            elif idx == mesh.ref_node + 1:  # Node above reference node (south node is reference)
                mesh.ref_pressure_source[idx] = mesh.a_pressure[idx, 4] * mesh.ref_node_value
                mesh.a_pressure[idx, 4] = 0
            elif idx == mesh.ref_node - mesh.ygrid:  # Node to the left of reference node (east node is reference)
                mesh.ref_pressure_source[idx] = mesh.a_pressure[idx, 1] * mesh.ref_node_value
                mesh.a_pressure[idx, 1] = 0
            elif idx == mesh.ref_node + mesh.ygrid:  # Node to the right of reference node (west node is reference)
                mesh.ref_pressure_source[idx] = mesh.a_pressure[idx, 3] * mesh.ref_node_value
                mesh.a_pressure[idx, 3] = 0


def pressure_correction_solver(mesh):
    # Format diagonals for A matrix
    if mesh.is_ref_node:  # If reference node is defined remove un-needed equations from system
        upper_off_diag = np.delete(-mesh.a_pressure[0:mesh.num_nodes - mesh.ygrid, 1], mesh.ref_node, 0)
        lower_off_diag = np.delete(-mesh.a_pressure[mesh.ygrid:, 3], mesh.ref_node - mesh.ygrid, 0)
        diag = np.delete(mesh.a_pressure[:, 0], mesh.ref_node, 0)
        super_diag = np.delete(-mesh.a_pressure[0:-1, 2], mesh.ref_node, 0)
        sub_diag = np.delete(-mesh.a_pressure[1:, 4], mesh.ref_node - 1, 0)
        b_matrix = np.delete(mesh.pressure_source[:] + mesh.ref_pressure_source[:], mesh.ref_node, 0)
    else:
        upper_off_diag = -mesh.a_pressure[0:mesh.num_nodes - mesh.ygrid, 1]
        lower_off_diag = -mesh.a_pressure[mesh.ygrid:, 3]
        diag = mesh.a_pressure[:, 0]
        super_diag = -mesh.a_pressure[0:-1, 2]
        sub_diag = -mesh.a_pressure[1:, 4]
        b_matrix = mesh.pressure_source[:]

    # Form A matrix
    diagonals = [diag, super_diag, sub_diag, upper_off_diag, lower_off_diag]
    a_sparse_matrix = diags(diagonals, [0, 1, -1, mesh.ygrid, -mesh.ygrid], format='csc')

    # Solve system
    # P = spla.spilu(a_sparse_matrix)
    # M_x = lambda xx: P.solve(xx)
    # M = spla.LinearOperator((len(b_matrix), len(b_matrix)), M_x)
    # p_correction_solved, exit_code = gmres(a_sparse_matrix, b_matrix, tol=1, M=M)
    p_correction_solved = spla.splu(a_sparse_matrix).solve(b_matrix)

    # If reference node defined insert 0 pressure correction value for this node to maintain consistency
    if mesh.is_ref_node:
        p_correction_solved = np.insert(p_correction_solved, mesh.ref_node, 0)
    mesh.pressure_correction = p_correction_solved


def correct_nodal_velocities(mesh, uv_relax):
    for idx in range(mesh.num_nodes):
        if idx < mesh.ygrid:  # Left boundary
            mesh.vel_correction[idx, 0] = 1 / mesh.a_momentum[idx, 0] * \
                                          (2 * mesh.pressure_correction_boundaries[idx, 0]
                                           - mesh.pressure_correction[idx + mesh.ygrid] -
                                           mesh.pressure_correction[idx]) / mesh.areas[idx, 1] / 2 * mesh.volumes[idx]
        elif idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary
            mesh.vel_correction[idx, 0] = 1 / mesh.a_momentum[idx, 0] \
                                          * (mesh.pressure_correction[idx - mesh.ygrid] + mesh.pressure_correction[idx]
                                             - 2 * mesh.pressure_correction_boundaries[idx % mesh.ygrid, 2]) \
                                          / mesh.areas[idx, 1] / 2 * mesh.volumes[idx]
        else:
            mesh.vel_correction[idx, 0] = 1 / mesh.a_momentum[idx, 0] \
                                          * (mesh.pressure_correction[idx - mesh.ygrid]
                                             - mesh.pressure_correction[idx + mesh.ygrid]) \
                                          / mesh.areas[idx, 1] / 2 * mesh.volumes[idx]
        if idx % mesh.ygrid == 0:  # Bottom boundary
            mesh.vel_correction[idx, 1] = 1 / mesh.a_momentum[idx, 0] * \
                                          (2 * mesh.pressure_correction_boundaries[mesh.column[idx], 3]
                                           - mesh.pressure_correction[idx + 1] - mesh.pressure_correction[idx]) \
                                          / mesh.areas[idx, 0] / 2 * mesh.volumes[idx]
        elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary
            mesh.vel_correction[idx, 1] = 1 / mesh.a_momentum[idx, 0] * \
                                          (mesh.pressure_correction[idx - 1] + mesh.pressure_correction[idx] -
                                           2 * mesh.pressure_correction_boundaries[mesh.column[idx], 1]) \
                                          / mesh.areas[idx, 0] / 2 * mesh.volumes[idx]
        else:
            mesh.vel_correction[idx, 1] = 1 / mesh.a_momentum[idx, 0] * \
                                          (mesh.pressure_correction[idx - 1] - mesh.pressure_correction[idx + 1]) \
                                          / mesh.areas[idx, 0] / 2 * mesh.volumes[idx]

    # Correct nodal velocity values
    mesh.vel[:, 0] += mesh.vel_correction[:, 0]
    mesh.vel[:, 1] += mesh.vel_correction[:, 1]


def correct_pressure(mesh, p_relaxation):
    if mesh.is_ref_node:
        normalized_correct = mesh.pressure_correction
    else:
        normalized_correct = mesh.pressure_correction - mesh.pressure_correction[mesh.ygrid + 1]
    correction = (p_relaxation * normalized_correct).reshape(mesh.num_nodes, 1)
    mesh.pressure = mesh.pressure + correction


def correct_face_velocities(mesh, uv_relax):
    for idx in range(mesh.num_nodes):
        if idx == mesh.num_nodes - 1:  # Top right corner
            continue
        elif idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary adjacent
            mesh.vel_face_correction[idx, 1] = (1 / mesh.a_momentum[idx + 1, 0] + 1 / mesh.a_momentum[idx, 0]) * \
                                               (mesh.pressure_correction[idx]
                                                - mesh.pressure_correction[idx + 1]) * mesh.areas[idx, 1]
        elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary adjacent
            mesh.vel_face_correction[idx, 0] = (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                idx, 0]) * (mesh.pressure_correction[idx]
                            - mesh.pressure_correction[idx + mesh.ygrid]) * mesh.areas[idx, 0]
        else:
            mesh.vel_face_correction[idx, 0] = (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                idx, 0]) * (mesh.pressure_correction[idx]
                            - mesh.pressure_correction[idx + mesh.ygrid]) * mesh.areas[idx, 0]

            mesh.vel_face_correction[idx, 1] = (1 / mesh.a_momentum[idx + 1, 0] + 1 / mesh.a_momentum[idx, 0]) * \
                                               (mesh.pressure_correction[idx]
                                                - mesh.pressure_correction[idx + 1]) * mesh.areas[idx, 1]
        if mesh.num_nodes - mesh.ygrid > idx:
            mesh.vel_face_correction[idx + mesh.ygrid, 2] = mesh.vel_face_correction[idx, 0]
        if idx % mesh.ygrid != (mesh.ygrid - 1):
            mesh.vel_face_correction[idx + 1, 3] = mesh.vel_face_correction[idx, 1]

    # Correct face velocity values
    mesh.vel_face[:, 0] += mesh.vel_face_correction[:, 0]
    mesh.vel_face[:, 1] += mesh.vel_face_correction[:, 1]
    mesh.vel_face[:, 2] += mesh.vel_face_correction[:, 2]
    mesh.vel_face[:, 3] += mesh.vel_face_correction[:, 3]


def pressure_extrapolation(mesh):
    # PRESSURE TO BOUNDARY
    # Left boundary
    mesh.pressure_boundaries[:, 0:1] = mesh.pressure[0:mesh.ygrid] - (mesh.dx[0] / mesh.dx[1]) * \
                                       (mesh.pressure[mesh.ygrid:2 * mesh.ygrid] - mesh.pressure[0:mesh.ygrid])
    # Top boundary
    mesh.pressure_boundaries[:, 1:2] = mesh.pressure[mesh.ygrid - 1::mesh.ygrid] - (mesh.dy[-1] / mesh.dy[-2]) * \
                                       (mesh.pressure[mesh.ygrid - 2::mesh.ygrid]
                                        - mesh.pressure[mesh.ygrid - 1::mesh.ygrid])
    # Right boundary
    mesh.pressure_boundaries[:, 2:3] = mesh.pressure[mesh.num_nodes - mesh.ygrid:] - (mesh.dx[-1] / mesh.dx[-2]) * \
                                       (mesh.pressure[mesh.num_nodes - 2 * mesh.ygrid:mesh.num_nodes - mesh.ygrid] -
                                        mesh.pressure[mesh.num_nodes - mesh.ygrid:])
    # Bottom boundary
    mesh.pressure_boundaries[:, 3:4] = mesh.pressure[0::mesh.ygrid] - (mesh.dy[0] / mesh.dy[1]) * \
                                       (mesh.pressure[1::mesh.ygrid] - mesh.pressure[0::mesh.ygrid])

    # PRESSURE CORRECTION TO BOUNDARY
    # Left boundary
    mesh.pressure_correction_boundaries[:, 0:1] = (mesh.pressure_correction[0:mesh.ygrid] -
                                                   (mesh.dx[0] / mesh.dx[1]) *
                                                   (mesh.pressure_correction[mesh.ygrid:2 * mesh.ygrid] -
                                                    mesh.pressure_correction[0:mesh.ygrid])).reshape(mesh.ygrid, 1)
    # Top boundary
    mesh.pressure_correction_boundaries[:, 1:2] = (mesh.pressure_correction[mesh.ygrid - 1::mesh.ygrid] -
                                                   (mesh.dy[-1] / mesh.dy[-2]) *
                                                   (mesh.pressure_correction[mesh.ygrid - 2::mesh.ygrid]
                                                    - mesh.pressure_correction[mesh.ygrid - 1::mesh.ygrid])
                                                   ).reshape(mesh.ygrid, 1)
    # Right boundary
    mesh.pressure_correction_boundaries[:, 2:3] = (mesh.pressure_correction[mesh.num_nodes - mesh.ygrid:] -
                                                   (mesh.dx[-1] / mesh.dx[-2]) *
                                                   (mesh.pressure_correction[mesh.num_nodes -
                                                                             2 * mesh.ygrid:mesh.num_nodes - mesh.ygrid] -
                                                    mesh.pressure_correction[mesh.num_nodes - mesh.ygrid:])).reshape(
        mesh.ygrid, 1)
    # Bottom boundary
    mesh.pressure_correction_boundaries[:, 3:4] = (mesh.pressure_correction[0::mesh.ygrid] -
                                                   (mesh.dy[0] / mesh.dy[1]) *
                                                   (mesh.pressure_correction[1::mesh.ygrid] -
                                                    mesh.pressure_correction[0::mesh.ygrid])).reshape(mesh.ygrid, 1)


def save_mesh_data(mesh, idx):
    np.save('data_live\\pressure_iteration_{}.npy'.format(idx), mesh.pressure)
    np.save('data_live\\u_vel_iteration_{}.npy'.format(idx), mesh.vel[:, 0])
    np.save('data_live\\v_vel_iteration_{}.npy'.format(idx), mesh.vel[:, 1])


def save_all_data(mesh, mass_imbalance, residuals_u, residuals_v):
    t = time.time()
    Path("complete_data\\data_{}".format(t)).mkdir(exist_ok=True)
    for filename in os.listdir('data_live\\'):
        shutil.move('data_live\\{}'.format(filename), 'complete_data\\data_{}\\'.format(t))
    np.save('complete_data\\data_{}\\x_coordinates.npy'.format(t), mesh.x_coordinates)
    np.save('complete_data\\data_{}\\y_coordinates.npy'.format(t), mesh.y_coordiantes)
    np.save('complete_data\\data_{}\\x_grid.npy'.format(t), mesh.xgrid)
    np.save('complete_data\\data_{}\\y_grid.npy'.format(t), mesh.ygrid)
    np.save('complete_data\\data_{}\\residual_u.npy'.format(t), residuals_u)
    np.save('complete_data\\data_{}\\residual_v.npy'.format(t), residuals_v)
    np.save('complete_data\\data_{}\\residual_mi.npy'.format(t), mass_imbalance)
    np.save('complete_data\\data_{}\\pressure_final.npy'.format(t), mesh.pressure)
    np.save('complete_data\\data_{}\\u_vel_final.npy'.format(t), mesh.vel[:, 0])
    np.save('complete_data\\data_{}\\v_vel_final.npy'.format(t), mesh.vel[:, 1])
    if mesh.is_object:
        np.save('complete_data\\data_{}\\obj_position.npy'.format(t), mesh.obj_p)
        np.save('complete_data\\data_{}\\obj_size.npy'.format(t), mesh.obj_s)
        np.save('complete_data\\data_{}\\domain_x.npy'.format(t), mesh.lx)
        np.save('complete_data\\data_{}\\domain_y.npy'.format(t), mesh.ly)
    return t


def visualize(mesh, fig1, ax1, ax2):
    # Pressure contour with velocity streamlines
    x = mesh.x_coordinates
    y = mesh.y_coordiantes
    pressure = mesh.pressure.reshape((mesh.ygrid, mesh.xgrid), order='F')
    plt.figure(fig1.number)
    ax1.cla()
    ax1.contourf(x, y, pressure)
    cp = plt.contourf(x, y, pressure)
    fig1.colorbar(mappable=cp, cax=ax2, orientation='horizontal')
    # Streamline plot
    u_vel = mesh.vel[:, 0].reshape((mesh.ygrid, mesh.xgrid), order='F')
    v_vel = mesh.vel[:, 1].reshape((mesh.ygrid, mesh.xgrid), order='F')
    ax1.streamplot(x, y, u_vel, v_vel, color='k', density=0.5)
    fig1.canvas.draw()
    fig1.canvas.flush_events()


def solution_convergence(err_tols, u_residuals, v_residuals, mass_imb):
    cvg = False
    if u_residuals[-1] / u_residuals[0] < err_tols and v_residuals[-1] / v_residuals[0] < err_tols and \
            mass_imb[-1] / mass_imb[0] < err_tols:
        cvg = True
    print(u_residuals[-1] / u_residuals[0], v_residuals[-1] / v_residuals[0],  mass_imb[-1] / mass_imb[0])
    return cvg


def fvm_solver(mesh, uv_relax, p_relax, max_iter, err_tols, visualize_on, save_data_on):
    mass_imbalance = []
    residuals_u = []
    residuals_v = []
    if visualize_on:
        plt.ion()
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.7])
        ax2 = fig1.add_axes([0.1, 0.85, 0.8, 0.05])
        x = mesh.x_coordinates
        y = mesh.y_coordiantes
        pressure = mesh.pressure.reshape((mesh.ygrid, mesh.xgrid), order='F')
        u_vel = mesh.vel[:, 0].reshape((mesh.ygrid, mesh.xgrid), order='F')
        v_vel = mesh.vel[:, 1].reshape((mesh.ygrid, mesh.xgrid), order='F')
        cp = plt.contourf(x, y, pressure)
        c_bar = fig1.colorbar(mappable=cp, cax=ax2, orientation = 'horizontal')
        c_bar.set_label('Pressure')
        ax1.streamplot(x, y, u_vel, v_vel, color='k', density=0.5)
        ax1.set_xlabel('$x$ [m]')
        ax1.set_ylabel('$y$ [m]', rotation=0)
        ax1.set_title('Pressure and Velocity Fields')
    set_boundary_values(mesh)
    for idx2 in range(max_iter):
        cvg = False
        momentum_formulation(mesh)
        residual_u, residual_v = momentum_solver(mesh, uv_relax)
        face_velocities(mesh, uv_relax)
        pressure_correction_formulation(mesh, uv_relax)
        pressure_correction_solver(mesh)
        correct_pressure(mesh, p_relax)
        pressure_extrapolation(mesh)
        set_boundary_values(mesh)
        correct_nodal_velocities(mesh, uv_relax)
        correct_face_velocities(mesh, uv_relax)
        if idx2 >= 1:
            residuals_u = np.append(residuals_u, np.linalg.norm(residual_u))
            residuals_v = np.append(residuals_v, np.linalg.norm(residual_v))
            mass_imbalance = np.append(mass_imbalance, np.linalg.norm(mesh.pressure_source))
            cvg = solution_convergence(err_tols, residuals_u, residuals_v, mass_imbalance)
        if idx2 % 5 == 0:
            if visualize_on:
                visualize(mesh, fig1, ax1, ax2)
            if save_data_on:
                save_mesh_data(mesh, idx2)
        if cvg and idx2 > 10:
            break
    return mesh, mass_imbalance, residuals_u, residuals_v


def main():
    problem = 1
    if problem == 1:
        ### LID CAVITY FLOW PROBLEM ###
        ## Constants and input parameters ##
        reynolds = 100
        u_top = 1
        p_top = 1
        uv_relax = 0.9
        p_relax = 0.1
        reference_node = [78, 40, 0]  # Column, row and value of pressure reference node

        ## Algorithm stopping criteria ##
        max_iter = 1500
        err_tols = 10 ** (-40)

        ## Solver preferences ##
        visualize_on = True
        save_data_on = False

        ## PROBLEM DEFINITION ##
        ## Create boundaries ##
        boundary_left_u = Boundary('D', 0, 'left')
        boundary_top_u = Boundary('D', u_top, 'top')
        boundary_right_u = Boundary('D', 0, 'right')
        boundary_bottom_u = Boundary('D', 0, 'bottom')
        boundary_left_v = Boundary('D', 0, 'left')
        boundary_top_v = Boundary('D', 0, 'top')
        boundary_right_v = Boundary('D', 0, 'right')
        boundary_bottom_v = Boundary('D', 0, 'bottom')
        boundary_left_p = Boundary('N', 0, 'left')
        boundary_top_p = Boundary('D', p_top, 'top')
        boundary_right_p = Boundary('N', 0, 'right')
        boundary_bottom_p = Boundary('N', 0, 'bottom')
        object = False

        ## Boundary set for domain ##
        boundaries_u = [boundary_left_u, boundary_right_u, boundary_top_u, boundary_bottom_u]
        boundaries_v = [boundary_left_v, boundary_right_v, boundary_top_v, boundary_bottom_v]
        boundaries_p = [boundary_left_p, boundary_right_p, boundary_top_p, boundary_bottom_p]

        ## Create meshed domain ##
        # mesh1 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 257, 257, 1, 1)
        # mesh1 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 129, 129, 1, 1)
        mesh1 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 80, 80, 1, 1)
        # Define space parameters
        mesh1.set_re(reynolds)
        mesh1.set_reference_node(reference_node)

        # Solve problem
        mesh1_solved, mass_imbalance, residuals_u, residuals_v = fvm_solver(mesh1, uv_relax, p_relax, max_iter,
                                                                            err_tols, visualize_on, save_data_on)
        # Save data if specified
        if save_data_on:
            folder = save_all_data(mesh1_solved, mass_imbalance, residuals_u, residuals_v)
            print(folder)
        print('Done problem 1')

    elif problem == 2:
        ### LID DRIVEN CAVITY WITH STEP PROBLEM ###
        ## Constants and input parameters ##
        reynolds = 200
        u_top = 1
        p_top = 1
        uv_relax = 0.8
        p_relax = 0.2
        reference_node = [25, 25, 0]  # Column, row and value of pressure reference node

        max_iter = 2000
        err_tols = 10 ** (-40)

        obj_size = [1 / 3, 1 / 3]
        obj_position = [0, 0]

        ## Create boundaries ##
        boundary_left_u = Boundary('D', 0, 'left')
        boundary_top_u = Boundary('D', u_top, 'top')
        boundary_right_u = Boundary('D', 0, 'right')
        boundary_bottom_u = Boundary('D', 0, 'bottom')
        boundary_left_v = Boundary('D', 0, 'left')
        boundary_top_v = Boundary('D', 0, 'top')
        boundary_right_v = Boundary('D', 0, 'right')
        boundary_bottom_v = Boundary('D', 0, 'bottom')
        boundary_left_p = Boundary('N', 0, 'left')
        boundary_top_p = Boundary('D', p_top, 'top')
        boundary_right_p = Boundary('N', 0, 'right')
        boundary_bottom_p = Boundary('N', 0, 'bottom')
        # Boundary set for domain
        boundaries_u = [boundary_left_u, boundary_right_u, boundary_top_u, boundary_bottom_u]
        boundaries_v = [boundary_left_v, boundary_right_v, boundary_top_v, boundary_bottom_v]
        boundaries_p = [boundary_left_p, boundary_right_p, boundary_top_p, boundary_bottom_p]

        mesh2 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 320, 320, 1, 1)
        # mesh2 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 129, 129, 1, 1)
        # mesh2 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 50, 50, 1, 1)
        mesh2.set_re(reynolds)
        mesh2.create_object(obj_size, obj_position)
        mesh2.set_reference_node(reference_node)
        mesh2_solved, mass_imbalance2, residuals_u2, residuals_v2 = fvm_solver(mesh2, uv_relax, p_relax,
                                                                               max_iter, err_tols)
        folder2 = save_all_data(mesh2_solved, mass_imbalance2, residuals_u2, residuals_v2)
        print(folder2)

    elif problem == 3:
        ### BACK-STEP FLOW PROBLEM ###
        ## Constants and input parameters ##
        reynolds = 100
        u_in = 1
        p_out = 0
        uv_relax = 0.8
        p_relax = 0.2
        reference_node = [25, 25, 0]  # Column, row and value of pressure reference node

        max_iter = 3000
        err_tols = 10 ** (-40)

        obj_size = [1 / 3, 1 / 3]
        obj_position = [0, 0]

        ## Create boundaries ##
        boundary_left_u = Boundary('D', u_in, 'left')
        boundary_top_u = Boundary('D', 0, 'top')
        # boundary_right_u = Boundary('None', 0, 'right')
        boundary_bottom_u = Boundary('D', 0, 'bottom')
        boundary_left_v = Boundary('D', 0, 'left')
        boundary_top_v = Boundary('D', 0, 'top')
        boundary_right_v = Boundary('D', 0, 'right')
        boundary_bottom_v = Boundary('D', 0, 'bottom')
        boundary_left_p = Boundary('N', 0, 'left')
        boundary_top_p = Boundary('N', 0, 'top')
        boundary_right_p = Boundary('D', p_out, 'right')
        boundary_bottom_p = Boundary('N', 0, 'bottom')
        # Boundary set for domain
        boundaries_u = [boundary_left_u, boundary_top_u, boundary_bottom_u]
        boundaries_v = [boundary_left_v, boundary_right_v, boundary_top_v, boundary_bottom_v]
        boundaries_p = [boundary_left_p, boundary_right_p, boundary_top_p, boundary_bottom_p]

        # mesh2 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 320, 320, 1, 1)
        mesh2 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 50, 50, 1, 1)
        mesh2.set_re(reynolds)
        mesh2.create_object(obj_size, obj_position)
        mesh2.set_reference_node(reference_node)
        mesh2_solved, mass_imbalance2, residuals_u2, residuals_v2 = fvm_solver(mesh2, uv_relax, p_relax,
                                                                               max_iter, err_tols)
        folder2 = save_all_data(mesh2_solved, mass_imbalance2, residuals_u2, residuals_v2)
        print(folder2)


if __name__ == '__main__':
    main()
