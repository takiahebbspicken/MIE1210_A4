import time
import numpy as np
import os
import shutil
# import multiprocessing
import matplotlib.pyplot as plt
# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
from numba import jit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from pathlib import Path



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

    def generate_2d_uniform_mesh(self, x_points, y_points, length_x, length_y):
        d_x = length_x / x_points
        deltas_x = np.ones((x_points + 1, 1)) * d_x
        deltas_x[0] = d_x / 2
        deltas_x[-1] = d_x / 2
        d_y = length_y / y_points
        deltas_y = np.ones((y_points + 1, 1)) * d_y
        deltas_y[0] = d_y / 2
        deltas_y[-1] = d_y / 2
        self.xgrid = x_points
        self.ygrid = y_points
        self.lx = length_x
        self.ly = length_y
        self.dx = deltas_x
        self.dy = deltas_y
        self.num_nodes = (len(self.dx) - 1) * (len(self.dy) - 1)
        self.x_coordinates = np.cumsum(self.dx[1:-1])
        self.y_coordiantes = np.cumsum(self.dy[1:-1])
        self.column = [int(np.floor(i / self.ygrid)) for i in range(self.num_nodes)]
        self.vel = np.zeros((self.num_nodes, 2))  # [u, v]
        # self.u_vel_boundaries = np.zeros((self.ygrid + 2, 4))
        # self.v_vel_boundaries = np.zeros((self.ygrid + 2, 4))
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
        self.vel_face_correction = np.zeros((self.num_nodes, 4)) # [E N W S]
        self.vel_face_correction_boundaries = np.zeros((self.num_nodes, 4))
        self.a_momentum = np.zeros((self.num_nodes, 5))  # [P E N W S]
        self.momentum_source_u = np.zeros((self.num_nodes, 2))  # [Su_u Su_v]
        self.momentum_source_pp = np.zeros((self.num_nodes, 2))  # [Spp_u Spp_v]
        self.momentum_pressure_source = np.zeros((self.num_nodes, 2))  # [S_pp_X, S_pp_Y]
        self.a_pressure = np.zeros((self.num_nodes, 5))  # [P E N W S]
        self.pressure_source = np.zeros((self.num_nodes, 1))
        self.a_pressure_boundary = np.ones((self.num_nodes, 4))  # 1 if node is not next to D u/v boundary, else 0 [E N W S]

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
        pass

    def set_rho(self, rhos):
        self.rhos = rhos

    def set_mus(self, mus):
        self.mus = mus

    def set_gamma(self, gammas):
        self.gammas = gammas


# Mesh Functions
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
    if mesh.u_bottom_boundary.type == 'D':
        mesh.u_vel_boundaries[0:-1:mesh.ygrid, 3] = mesh.u_bottom_boundary.value
    if mesh.u_top_boundary.type == 'D':
        mesh.u_vel_boundaries[mesh.ygrid - 1::mesh.ygrid, 1] = mesh.u_top_boundary.value

    # V boundaries
    if mesh.v_left_boundary.type == 'D':
        mesh.v_vel_boundaries[0:mesh.ygrid, 0] = mesh.v_left_boundary.value
    if mesh.v_right_boundary.type == 'D':
        mesh.v_vel_boundaries[mesh.num_nodes - mesh.ygrid:, 0] = mesh.v_right_boundary.value
    if mesh.v_bottom_boundary.type == 'D':
        mesh.vel_face[0:-1:mesh.ygrid, 3] = mesh.v_bottom_boundary.value
        mesh.v_vel_boundaries[0:-1:mesh.ygrid, 3] = mesh.v_bottom_boundary.value
        mesh.a_pressure_boundary[0:-1:mesh.ygrid, 3] = 0
    if mesh.v_top_boundary.type == 'D':
        mesh.vel_face[mesh.ygrid - 1::mesh.ygrid, 1] = mesh.v_top_boundary.value
        mesh.v_vel_boundaries[mesh.ygrid - 1::mesh.ygrid, 1] = mesh.v_top_boundary.value
        mesh.a_pressure_boundary[mesh.ygrid - 1::mesh.ygrid, 1] = 0

    if mesh.p_left_boundary.type == 'N':
        mesh.pressure_boundaries[:, 0:1] = mesh.pressure[0:mesh.ygrid] - \
                                                    mesh.p_left_boundary.value * mesh.dx[0]
    if mesh.p_right_boundary.type == 'N':
        mesh.pressure_boundaries[:, 2:3] = mesh.pressure[mesh.num_nodes - mesh.ygrid:] - \
                                                    mesh.p_left_boundary.value * mesh.dx[-1]
    if mesh.p_bottom_boundary.type == 'N':
        mesh.pressure_boundaries[:, 3:4] = mesh.pressure[0:-1:mesh.ygrid] - \
                                                    mesh.p_left_boundary.value * mesh.dy[0]
    if mesh.p_top_boundary.type == 'N':
        mesh.pressure_boundaries[:, 1:2] = mesh.pressure[mesh.ygrid - 1::mesh.ygrid] - \
                                                    mesh.p_left_boundary.value * mesh.dy[-1]

# @jit(parallel=True)
def momentum_formulation(mesh):
    for idx in range(mesh.num_nodes):
        _, dpx_p, _, _, dpy_p, _ = pressure_derivatives(mesh, idx)
        f_e = mesh.rhos * mesh.vel_face[idx, 0] * mesh.areas[idx, 0]
        f_n = mesh.rhos * mesh.vel_face[idx, 1] * mesh.areas[idx, 1]
        d_e = mesh.mus * mesh.areas[idx, 0] / mesh.dx[mesh.column[idx] + 1]
        d_n = mesh.mus * mesh.areas[idx, 1] / mesh.dy[(idx - mesh.column[idx] * mesh.ygrid) + 1]
        mesh.a_momentum[idx, 1] = (np.abs(f_e) - f_e) / 2 + d_e
        mesh.a_momentum[idx, 2] = (np.abs(f_n) - f_n) / 2 + d_n
        if idx == 0:  # Bottom left boundary corner
            f_w = mesh.rhos * mesh.vel_face[idx, 2] * mesh.areas[idx, 0]
            f_s = mesh.rhos * mesh.vel_face[idx, 3] * mesh.areas[idx, 1]
            d_w = mesh.mus * mesh.areas[idx, 0] / mesh.dx[mesh.column[idx]]
            d_s = mesh.mus * mesh.areas[idx, 1] / mesh.dy[(idx - mesh.column[idx] * mesh.ygrid)]
            mesh.a_momentum[idx, 3] = (np.abs(f_w) + f_w) / 2 + d_w
            mesh.a_momentum[idx, 4] = (np.abs(f_s) + f_s) / 2 + d_s
        elif idx < mesh.ygrid:  # Left boundary
            f_w = mesh.rhos * mesh.vel_face[idx, 2] * mesh.areas[idx, 0]
            d_w = mesh.mus * mesh.areas[idx, 0] / mesh.dx[mesh.column[idx]]
            mesh.a_momentum[idx, 3] = (np.abs(f_w) + f_w) / 2 + d_w
        elif idx % mesh.ygrid == 0 and idx != 0:  # Bottom boundary
            f_s = mesh.rhos * mesh.vel_face[idx, 3] * mesh.areas[idx, 1]
            d_s = mesh.mus * mesh.areas[idx, 1] / mesh.dy[(idx - mesh.column[idx] * mesh.ygrid)]
            mesh.a_momentum[idx, 4] = (np.abs(f_s) + f_s) / 2 + d_s
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
        # Assign a_w and a_s for future nodes
        if idx < mesh.num_nodes - mesh.ygrid:
            mesh.a_momentum[idx + mesh.ygrid, 3] = mesh.a_momentum[idx, 1]
        if idx % mesh.ygrid != (mesh.ygrid - 1):
            mesh.a_momentum[idx + 1, 4] = mesh.a_momentum[idx, 2]
        mesh.momentum_pressure_source[idx, 0] = - dpx_p * mesh.volumes[idx]  # TODO: verify sign here and below
        mesh.momentum_pressure_source[idx, 1] = - dpy_p * mesh.volumes[idx]


def momentum_solver(mesh):
    upper_off_diag = -mesh.a_momentum[0:mesh.num_nodes - mesh.ygrid, 1]
    lower_off_diag = -mesh.a_momentum[mesh.ygrid:, 3]
    diag = mesh.a_momentum[:, 0]
    super_diag = -mesh.a_momentum[0:-1, 2]
    sub_diag = -mesh.a_momentum[1:, 4]
    diagonals = [diag, super_diag, sub_diag, upper_off_diag, lower_off_diag]
    a_sparse_matrix = diags(diagonals, [0, 1, -1, mesh.ygrid, -mesh.ygrid], format='csc')
    b_matrix_u = mesh.momentum_pressure_source[:, 0] + mesh.momentum_source_u[:, 0]
    b_matrix_v = mesh.momentum_pressure_source[:, 1] + mesh.momentum_source_u[:, 1]
    u_solved = spsolve(a_sparse_matrix, b_matrix_u)
    v_solved = spsolve(a_sparse_matrix, b_matrix_v)
    mesh.vel[:, 0] = u_solved
    mesh.vel[:, 1] = v_solved


# @jit(nopython=True, parallel=True)
def face_velocities(mesh):
    # Nodes not on boundaries
    for idx in range(mesh.num_nodes - mesh.ygrid):
        dpx_E, dpx_p, dpx_e, dpy_N, dpy_p, dpy_n = pressure_derivatives(mesh, idx)
        if idx >= mesh.ygrid * mesh.xgrid - mesh.ygrid:  # Right boundary adjacent
            continue
        elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary adjacent
            continue
        else:
            mesh.vel_face[idx, 0] = (mesh.vel[idx, 0] + mesh.vel[idx + mesh.ygrid, 0]) / 2 \
                                    + (1 / 2) * (mesh.volumes[idx] / mesh.a_momentum[idx, 0] * dpx_p
                                                 + mesh.volumes[idx + mesh.ygrid] /
                                                 mesh.a_momentum[idx + mesh.ygrid, 0] * dpx_E) - \
                                    (mesh.volumes[idx] / 2 + mesh.volumes[idx + mesh.ygrid] / 2) * \
                                    (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                                        idx + mesh.ygrid, 0]) \
                                    * dpx_e
            mesh.vel_face[idx, 1] = (mesh.vel[idx, 1] + mesh.vel[idx + mesh.ygrid, 1]) / 2 \
                                    + (1 / 2) * (mesh.volumes[idx] / mesh.a_momentum[idx, 0] * dpy_p
                                                 + mesh.volumes[idx + mesh.ygrid] /
                                                 mesh.a_momentum[idx + mesh.ygrid, 0] * dpy_N) - \
                                    (mesh.volumes[idx] / 2 + mesh.volumes[idx + mesh.ygrid] / 2) * \
                                    (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                                        idx + mesh.ygrid, 0]) \
                                    * dpy_n
        # Assign w and s face velocities
        mesh.vel_face[idx + mesh.ygrid, 2] = mesh.vel_face[idx, 0]
        mesh.vel_face[idx + 1, 3] = mesh.vel_face[idx, 1]


# @jit()
def pressure_derivatives(mesh, idx):  # TODO: pressure_boundary variable and check top calculations
    _ = None
    if idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary
        if idx == mesh.num_nodes - 1:  # Top right corner
            dpx_p = (2 * mesh.pressure_boundaries[idx - mesh.ygrid * mesh.column[idx], 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        elif idx == mesh.num_nodes - mesh.ygrid:  # Bottom right corner
            dpx_p = (2 * mesh.pressure_boundaries[idx - mesh.ygrid * mesh.column[idx], 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 3] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        else:
            dpx_p = (2 * mesh.pressure_boundaries[idx - mesh.ygrid * mesh.column[idx], 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        return _, dpx_p, _, _, dpy_p, _
    elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary
        if idx == mesh.num_nodes - 1:  # Top left corner
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2 * mesh.pressure_boundaries[idx, 0]) \
                    / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        else:
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        return _, dpx_p, _, _, dpy_p, _
    else:  # Added interpolations for face velocity calculations
        if idx == 0:  # Bottom left boundary corner adjacent
            dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) / \
                    (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2 * mesh.pressure_boundaries[idx, 0]) \
                    / (2 * mesh.areas[idx, 1])
            dpy_p = (mesh.pressure[idx + 1] + mesh.pressure[idx] - 2 * mesh.pressure_boundaries[idx, 3]) / (
                    2 * mesh.areas[idx, 0])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
            # elif idx > mesh.num_nodes - 2 * mesh.ygrid:
        elif idx < mesh.ygrid:  # Left boundary adjacent
            if idx < mesh.num_nodes - 2 * mesh.ygrid:
                dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) \
                        / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2
                     * mesh.pressure_boundaries[idx, 0]) \
                    / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        elif idx >= mesh.num_nodes - 2 * mesh.ygrid:  # Two from right boundary
            dpx_E = (2 * mesh.pressure_boundaries[idx % mesh.ygrid + 1, 2] -
                     mesh.pressure[idx] - mesh.pressure[idx + mesh.ygrid]) \
                    / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        elif idx % mesh.ygrid == 0 and idx != 0:  # Bottom boundary adjacent
            dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) \
                    / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpy_p = (mesh.pressure[idx + 1] + mesh.pressure[idx] -
                     2 * mesh.pressure_boundaries[mesh.column[idx], 3]) / (2 * mesh.areas[idx, 0])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        elif idx % mesh.ygrid == (mesh.ygrid - 2):  # Two from top boundary
            dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) \
                    / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpy_N = (2 * mesh.pressure_boundaries[mesh.column[idx], 1] - mesh.pressure[idx] - mesh.pressure[
                idx + 1]) / \
                    (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                    2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        else:
            dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 1])
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column[idx] + 1])
            dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
            dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (
                        2 * mesh.dy[idx - mesh.column[idx] * mesh.ygrid + 1])
        return dpx_E, dpx_p, dpx_e, dpy_N, dpy_p, dpy_n


def pressure_correction_formulation(mesh):  # TODO: MAYBE AN ERROR HERE
    f_e = np.zeros(mesh.num_nodes)
    f_w = np.zeros(mesh.num_nodes)
    f_n = np.zeros(mesh.num_nodes)
    f_s = np.zeros(mesh.num_nodes)
    for idx in range(mesh.num_nodes):
        f_e[idx] = mesh.rhos * mesh.vel_face[idx, 0] * mesh.areas[idx, 0]
        f_n[idx] = mesh.rhos * mesh.vel_face[idx, 1] * mesh.areas[idx, 1]
        f_w[idx] = mesh.rhos * mesh.vel_face[idx, 2] * mesh.areas[idx, 0]
        f_s[idx] = mesh.rhos * mesh.vel_face[idx, 3] * mesh.areas[idx, 1]
        if mesh.a_pressure_boundary[idx, 0] == 1:
            mesh.a_pressure[idx, 1] = (mesh.rhos * mesh.areas[idx, 0] ** 2 / 2) * (
                    1 / mesh.a_momentum[idx + mesh.ygrid, 0]
                    + 1 / mesh.a_momentum[idx, 0])
        else:
            mesh.a_pressure[idx, 1] = 0
        if mesh.a_pressure_boundary[idx, 1] == 1:
            mesh.a_pressure[idx, 2] = (mesh.rhos * mesh.areas[idx, 1] ** 2 / 2) * (1 / mesh.a_momentum[idx + 1, 0]
                                                                                   + 1 / mesh.a_momentum[idx, 0])
        else:
            mesh.a_pressure[idx, 2] = 0
        if mesh.a_pressure_boundary[idx, 2] == 1:
            mesh.a_pressure[idx, 3] = (mesh.rhos * mesh.areas[idx, 0] ** 2 / 2) * (
                    1 / mesh.a_momentum[idx - mesh.ygrid, 0]
                    + 1 / mesh.a_momentum[idx, 0])
        else:
            mesh.a_pressure[idx, 3] = 0
        if mesh.a_pressure_boundary[idx, 3] == 0:
            mesh.a_pressure[idx, 4] = (mesh.rhos * mesh.areas[idx, 1] ** 2 / 2) * (1 / mesh.a_momentum[idx - 1, 0]
                                                                                   + 1 / mesh.a_momentum[idx, 0])
        else:
            mesh.a_pressure[idx, 4] = 0
        mesh.pressure_source[idx] = f_w[idx] - f_e[idx] + f_s[idx] - f_n[idx]
        mesh.a_pressure[idx, 0] = mesh.a_pressure[idx, 1] + mesh.a_pressure[idx, 2] + mesh.a_pressure[idx, 3] + \
                                  mesh.a_pressure[idx, 4]


def pressure_correction_solver(mesh):
    upper_off_diag = -mesh.a_pressure[0:mesh.num_nodes - mesh.ygrid, 1]
    lower_off_diag = -mesh.a_pressure[mesh.ygrid:, 3]
    diag = mesh.a_pressure[:, 0]
    super_diag = -mesh.a_pressure[0:-1, 2]
    sub_diag = -mesh.a_pressure[1:, 4]
    diagonals = [diag, super_diag, sub_diag, upper_off_diag, lower_off_diag]
    a_sparse_matrix = diags(diagonals, [0, 1, -1, mesh.ygrid, -mesh.ygrid], format='csc')
    b_matrix = mesh.pressure_source[:]
    p_correction_solved = spsolve(a_sparse_matrix, b_matrix)
    mesh.pressure_source = p_correction_solved


def correct_nodal_velocities(mesh, u_relaxation, v_relaxation):  # TODO: this function
    for idx in range(mesh.num_nodes):
        if idx < mesh.ygrid:  # Left boundary
            mesh.vel_correction[idx, 0] = 1 / mesh.a_momentum[idx, 0] * \
                                          (2*mesh.pressure_correction_boundaries[idx, 0]
                                           - mesh.pressure_correction[idx + mesh.ygrid] -
                                           mesh.pressure_correction[idx]) / mesh.areas[idx, 1] / 2 * mesh.volumes[idx]
        elif idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary
            mesh.vel_correction[idx, 0] = 1 / mesh.a_momentum[idx, 0] \
                                          * (mesh.pressure_correction[idx - mesh.ygrid] + mesh.pressure_correction[idx]
                                             - 2*mesh.pressure_correction_boundaries[idx % mesh.ygrid, 2]) \
                                          / mesh.areas[idx, 1] / 2 * mesh.volumes[idx]
        else:
            mesh.vel_correction[idx, 0] = 1 / mesh.a_momentum[idx, 0] \
                                          * (mesh.pressure_correction[idx - mesh.ygrid]
                                             - mesh.pressure_correction[idx + mesh.ygrid]) \
                                          / mesh.areas[idx, 1] / 2 * mesh.volumes[idx]
        if idx % mesh.ygrid == 0 and idx != 0:  # Bottom boundary
            mesh.vel_correction[idx, 1] = 1 / mesh.a_momentum[idx, 0] * \
                                          (2 * mesh.pressure_correction_boundaries[mesh.column[idx], 3]
                                           - mesh.pressure_correction[idx + 1] - mesh.pressure_correction[idx]) \
                                          / mesh.areas[idx, 0] / 2 * mesh.volumes[idx]
        elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary
            mesh.vel_correction[idx, 1] = 1 / mesh.a_momentum[idx, 0] *\
                                          (mesh.pressure_correction[idx - 1] + mesh.pressure_correction[idx] -
                                           2 * mesh.pressure_correction_boundaries[mesh.column[idx], 1]) \
                                          / mesh.areas[idx, 0] / 2 * mesh.volumes[idx]
        else:
            mesh.vel_correction[idx, 1] = 1 / mesh.a_momentum[idx, 0] * \
                                          (mesh.pressure_correction[idx - 1] - mesh.pressure_correction[idx + 1]) \
                                          / mesh.areas[idx, 0] / 2 * mesh.volumes[idx]
    mesh.vel[:, 0] += u_relaxation * mesh.vel_correction[:, 0]
    mesh.vel[:, 1] += v_relaxation * mesh.vel_correction[:, 1]


def correct_pressure(mesh, p_relaxation):
    mesh.pressure[:] += p_relaxation * (mesh.pressure_correction[:] - mesh.pressure_correction[mesh.ygrid + 1])


def correct_face_velocities(mesh):
    for idx in range(mesh.num_nodes - mesh.ygrid):
        if idx >= mesh.ygrid * mesh.xgrid - mesh.ygrid:  # Right boundary adjacent
            continue
        elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary adjacent
            continue
        else:
            mesh.vel_face_correction[idx, 0] = (1/mesh.a_momentum[idx + mesh.ygrid, 0] + 1/mesh.a_momentum[idx, 0]) * \
                                               (mesh.pressure_correction[idx]
                                                - mesh.pressure_correction[idx + mesh.ygrid]) * mesh.areas[idx, 0]

            mesh.vel_face_correction[idx, 1] = (1/mesh.a_momentum[idx + 1, 0] + 1/mesh.a_momentum[idx, 0]) * \
                                               (mesh.pressure_correction[idx]
                                                - mesh.pressure_correction[idx + 1]) * mesh.areas[idx, 1]
        mesh.vel_face_correction[idx + mesh.ygrid, 2] = mesh.vel_face_correction[idx, 0]
        mesh.vel_face_correction[idx + 1, 3] = mesh.vel_face_correction[idx, 1]
        mesh.vel_face[:, 0] += mesh.vel_face_correction[idx, 0]
        mesh.vel_face[:, 1] += mesh.vel_face_correction[idx, 1]
        mesh.vel_face[:, 2] += mesh.vel_face_correction[idx, 2]
        mesh.vel_face[:, 3] += mesh.vel_face_correction[idx, 3]


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
                                     (mesh.pressure[mesh.num_nodes - 2* mesh.ygrid:mesh.num_nodes - mesh.ygrid] -
                                      mesh.pressure[mesh.num_nodes - mesh.ygrid:])
    # Bottom boundary
    mesh.pressure_boundaries[:, 3:4] = mesh.pressure[0:-1:mesh.ygrid] - (mesh.dy[0] / mesh.dy[1]) * \
                                     (mesh.pressure[1:-1:mesh.ygrid] - mesh.pressure[0:-1:mesh.ygrid])

    # PRESSURE CORRECTION TO BOUNDARY
    # Left boundary
    mesh.pressure_correction_boundaries[:, 0:1] = mesh.pressure_correction[0:mesh.ygrid] - \
                                                (mesh.dx[0] / mesh.dx[1]) * \
                                     (mesh.pressure_correction[mesh.ygrid:2 * mesh.ygrid] -
                                      mesh.pressure_correction[0:mesh.ygrid])
    # Top boundary
    mesh.pressure_correction_boundaries[:, 1:2] = mesh.pressure_correction[mesh.ygrid - 1::mesh.ygrid] - \
                                                (mesh.dy[-1] / mesh.dy[-2]) * \
                                     (mesh.pressure_correction[mesh.ygrid - 2::mesh.ygrid]
                                      - mesh.pressure_correction[mesh.ygrid - 1::mesh.ygrid])
    # Right boundary
    mesh.pressure_correction_boundaries[:, 2:3] = mesh.pressure_correction[mesh.num_nodes - mesh.ygrid:] - \
                                                (mesh.dx[-1] / mesh.dx[-2]) * \
                                     (mesh.pressure_correction[mesh.num_nodes -
                                                               2 * mesh.ygrid:mesh.num_nodes - mesh.ygrid] -
                                      mesh.pressure_correction[mesh.num_nodes - mesh.ygrid:])
    # Bottom boundary
    mesh.pressure_correction_boundaries[:, 3:4] = mesh.pressure_correction[0:-1:mesh.ygrid] - \
                                                (mesh.dy[0] / mesh.dy[1]) * \
                                     (mesh.pressure_correction[1:-1:mesh.ygrid] -
                                      mesh.pressure_correction[0:-1:mesh.ygrid])


# Save mesh data (pressure and velocity field)
def save_mesh_data(mesh, idx):
    np.save('data_live\\pressure_iteration_{}.npy'.format(idx), mesh.pressure)
    np.save('data_live\\vel_iteration_{}.npy'.format(idx), mesh.vel)


def save_all_data():
    t = time.time()
    Path("complete_data\\data_{}".format(t)).mkdir(exist_ok=True)
    Path("complete_plots\\plots_{}".format(t)).mkdir(exist_ok=True)
    for filename in os.listdir('data_live\\'):
        shutil.move(filename, 'complete_data\\data_{}\\'.format(t))
    for filename in os.listdir('plots_live\\'):
        shutil.move(filename, 'complete_data\\plots_{}\\'.format(t))


def visualize(mesh, errs_u, errs_v, errs_p, idx):
    fig1, ax1 = plt.subplots(1, 1)
    # Pressure contour
    x = mesh.x_coordinates
    y = mesh.y_coordiantes
    pressure = mesh.pressure.reshape((mesh.ygrid, mesh.xgrid))
    cp = ax1.contourf(x, y, pressure)
    c_bar = fig1.colorbar(cp)
    c_bar.set_label('Pressure', rotation=180)
    ax1.set_xlabel('$x$ [m]')
    ax1.set_ylabel('$y$ [m]', rotation=0)
    ax1.set_title('Pressure and Velocity Fields')
    plt.tight_layout()
    # Streamline plot
    vel_mag = np.sqrt(mesh.vel[:, 0]**2 + mesh.vel[:, 1]**2).reshape((mesh.ygrid, mesh.xgrid))
    plt.streamplot(x, y, mesh.vel[:, 0].reshape((mesh.ygrid, mesh.xgrid)),
                   mesh.vel[:, 1].reshape((mesh.ygrid, mesh.xgrid)), density=0.5, linewidth=vel_mag/vel_mag.max())
    if idx == 0:
        plt.show()
    else:
        plt.draw()

    # Convergence plot
    fig2, ax2 = plt.subplots(1, 1)
    x = np.arange(len(errs_u))
    y = errs_u
    x2 = np.arange(len(errs_v))
    y2 = errs_v
    x3 = np.arange(len(errs_p))
    y3 = errs_p
    ln1, = ax2.plot(x, y, label='u velocity error')
    ln2, = ax2.plot(x2, y2, label='v velocity error')
    ln3, = ax2.plot(x3, y3, label='pressure error')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Error Norms', rotation=0)
    ax2.set_title('Convergence')
    ax2.legend(handles=[ln1, ln2, ln3])
    plt.tight_layout()
    if idx == 0:
        plt.show()
    else:
        plt.draw()


def solution_convergence(mesh, pressure_old, vel_old, err_tols):
    cvg = False
    pressure_error = np.abs(mesh.pressure - pressure_old)
    vel_error = np.abs(mesh.vel - vel_old)
    u_error = vel_error[:, 0]
    v_error = vel_error[:, 1]
    if np.linalg.norm(u_error) < err_tols and np.linalg.norm(v_error) < err_tols and np.linalg.norm(pressure_error) < err_tols:
        cvg = True
    err_u, err_v, err_p = np.linalg.norm(u_error), np.linalg.norm(v_error), np.linalg.norm(pressure_error)
    return cvg, err_u, err_v, err_p


def fvm_solver(mesh, u_relax, v_relax, p_relax, max_iter, err_tols):
    errs_u = []
    errs_v = []
    errs_p = []
    for i in range(max_iter):
        cvg = False
        set_boundary_values(mesh)
        momentum_formulation(mesh)
        momentum_solver(mesh)
        face_velocities(mesh)
        pressure_correction_formulation(mesh)
        pressure_correction_solver(mesh)
        correct_nodal_velocities(mesh, u_relax, v_relax)
        correct_pressure(mesh, p_relax)
        correct_face_velocities(mesh)
        pressure_extrapolation(mesh)
        save_mesh_data(mesh, i)
        if i > 1:
            cvg, err_u, err_v, err_p = solution_convergence(mesh, pressure_old, vel_old, err_tols)
            errs_u = np.append(errs_u, err_u)
            errs_v = np.append(errs_v, err_v)
            errs_p = np.append(errs_p, err_p)
            visualize(mesh, errs_u, errs_v, errs_p, i)
        if cvg and i > 10:
            break
        pressure_old = mesh.pressure
        vel_old = mesh.pressure
    save_all_data()


def main():
    ### LID CAVITY FLOW PROBLEM ###
    ## Constants and input parameters ##
    u_top = 1
    p_top = 1
    u_relax = 0.2
    v_relax = 0.2
    p_relax = 0.2
    max_iter = 2
    err_tols = 10**(-1)

    object = False
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

    # Create Domain
    mesh1 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 10, 10, 1, 1)
    fvm_solver(mesh1, u_relax, v_relax, p_relax, max_iter, err_tols)


    ### LID DRIVEN CAVITY WITH STEP PROBLEM ###
    ## Constants and input parameters ##
    u_top = 1
    p_top = 1
    u_relax = 0.2
    v_relax = 0.2
    p_relax = 0.2
    max_iter = 100
    err_tols = 10 ** (-1)
    object = True
    size = 1
    position = 1

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
    mesh2 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 10, 10, 1, 1)
    mesh2.create_object(size, position)

    ### BACK-STEP FLOW PROBLEM ###
    ## Constants and input parameters ##
    u_top = 1
    p_top = 1
    u_relax = 0.2
    v_relax = 0.2
    p_relax = 0.2
    max_iter = 100
    err_tols = 10 ** (-1)
    object = True
    size = 1
    position = 1
    object = True
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
    mesh3 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 10, 10, 1, 1)
    mesh3.create_object(size, position)

if __name__ == '__main__':
    main()
