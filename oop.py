import time
import numpy as np
import pickle
import dill
# import multiprocessing
import matplotlib.pyplot as plt
# from multiprocessing import Pool
from functools import partial
# from pathos.multiprocessing import ProcessingPool as Pool
from numba import njit, jit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


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
        self.vel = np.zeros((self.num_nodes, 2))
        # self.u_vel_boundaries = np.zeros((self.ygrid + 2, 4))
        # self.v_vel_boundaries = np.zeros((self.ygrid + 2, 4))
        self.u_vel_boundaries = np.zeros((self.num_nodes, 4))  # [E N W S] boundary velocity conditions
        self.v_vel_boundaries = np.zeros((self.num_nodes, 4))  # [E N W S]
        self.vel_correction = np.zeros((self.num_nodes, 2))
        self.vel_correction_boundaries = np.zeros((self.ygrid + 2, 4))
        self.pressure = np.zeros((self.num_nodes, 1))
        self.pressure_boundaries = np.zeros((self.ygrid + 2, 4))
        self.pressure_correction = np.zeros((self.num_nodes, 1))
        self.pressure_correction_boundaries = np.zeros((self.ygrid + 2, 4))
        self.vel_face = np.zeros((self.num_nodes, 4))  # [E N W S]
        self.vel_face_boundaries = np.zeros((self.num_nodes, 4))
        self.vel_face_correction = np.zeros((self.num_nodes, 4))
        self.vel_face_correction_boundaries = np.zeros((self.ygrid + 2, 4))
        self.a_momentum = np.zeros((self.num_nodes, 5))  # [P E N W S]
        self.momentum_source_u = np.zeros((self.num_nodes, 2))  # [Su_u Su_v]
        self.momentum_source_pp = np.zeros((self.num_nodes, 2))  # [Spp_u Spp_v]
        self.momentum_pressure_source = np.zeros((self.num_nodes, 2))  # [S_pp_X, S_pp_Y]
        self.a_pressure = np.zeros((self.num_nodes, 5))  # [P E N W S]
        self.pressure_source = np.zeros((self.num_nodes, 1))
        self.a_pressure_boundary = np.ones((self.num_nodes, 1))  # 1 if node is not next to D u/v boundary, else 0

        # Maybe
        # self.vel = np.zeros(((x_points + 2) * (y_points + 2), 2))
        # self.vel_correction = np.zeros(((x_points + 2) * (y_points + 2), 2))
        # self.pressure = np.zeros(((x_points + 2) * (y_points + 2), 1))
        # self.pressure_correction = np.zeros(((x_points + 2) * (y_points + 2), 1))
        # self.vel_face = np.zeros(((x_points + 2) * (y_points + 2), 4))
        # self.vel_face_correction = np.zeros(((x_points + 2) * (y_points + 2), 4))
        # self.a_momentum = np.zeros(((x_points + 2) * (y_points + 2), 5))
        # self.momentum_source = np.zeros(((x_points + 2) * (y_points + 2), 2))
        # self.a_pressure = np.zeros(((x_points + 2) * (y_points + 2), 5))
        # self.pressure_source = np.zeros(((x_points + 2) * (y_points + 2), 2))

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

    def set_rho(self, rhos):
        self.rhos = rhos

    def set_mus(self, mus):
        self.mus = mus

    def set_gamma(self, gammas):
        self.gammas = gammas


class MomentumCoefficient:
    def __init__(self, mesh):
        self.mesh = mesh
        self.coefficients()

    def face_values(self, idx):
        rho_e = self.mesh.nodes.num_nodes
        rho_w = self.mesh.nodes.num_nodes
        rho_n = self.mesh.nodes.num_nodes
        rho_s = self.mesh.nodes.num_nodes
        u_e = self.mesh.nodes.num_nodes
        u_w = self.mesh.nodes.num_nodes
        v_n = self.mesh.nodes.num_nodes
        v_s = self.mesh.nodes.num_nodes
        gamma_e = self.mesh.nodes.num_nodes
        gamma_w = self.mesh.nodes.num_nodes
        gamma_n = self.mesh.nodes.num_nodes
        gamma_s = self.mesh.nodes.num_nodes
        self.delta_e = self.mesh.nodes.num_nodes
        self.delta_w = self.mesh.nodes.num_nodes
        self.delta_n = self.mesh.nodes.num_nodes
        self.delta_s = self.mesh.nodes.num_nodes
        self.area_e = self.mesh.nodes.num_nodes
        self.area_w = self.mesh.nodes.num_nodes
        self.area_n = self.mesh.nodes.num_nodes
        self.area_s = self.mesh.nodes.num_nodes
        self.f_e = rho_e * u_e
        self.f_w = rho_w * u_w
        self.f_n = rho_n * v_n
        self.f_s = rho_s * v_s
        self.d_e = gamma_e * self.area_e / self.delta_e
        self.d_w = gamma_w * self.area_w / self.delta_w
        self.d_n = gamma_n * self.area_n / self.delta_n
        self.d_s = gamma_s * self.area_s / self.delta_s

    def coefficients(self):
        # Interior nodes
        for i in range(self.mesh.ygrid, self.mesh.num_nodes - self.mesh.ygrid):
            if i % self.mesh.ygrid != (self.mesh.ygrid - 1) and i % self.mesh.ygrid != 0:
                self.face_values(i)
                self.a_p[i] = []
                self.a_e[i] = []
                self.a_w[i] = []
                self.a_n[i] = []
                self.a_s[i] = []
        # Left boundary

        # Right boundary

        # Top boundary

        # Bottom boundary


class Initialize:
    def __init__(self):
        pass


class Material:
    def __init__(self, gamma=1, rho=1, mu=1):
        self.set_properties(gamma, rho, mu)

    def set_properties(self, gamma, rho, mu):
        self.diffusion = gamma
        self.density = rho
        self.viscosity = mu


class LinearSystem:
    def __init__(self, coefficients, values):
        self.coefficients = coefficients
        self.values = values

    def form_amatrix(self):
        pass

    def form_bmatrix(self):
        pass


class SystemSolver:
    def __init__(self, lin_system):
        self.linear_system = lin_system


class Callback:
    def __init__(self):
        pass
    # If using iterative linear solver


class Visualization:
    def __init__(self, mesh):
        self.mesh = mesh
    # Streamline plot function
    # Contour plot function
    # Contour with streamlines function
    # Velocity profile along horizontal line through cavity center function


# Mesh Functions
def set_boundary_values(mesh):
    # U boundaries
    if mesh.u_left_boundary.type == 'D':
        mesh.vel_face[0:mesh.ygrid, 2] = mesh.u_left_boundary.value
        mesh.u_vel_boundaries[0:mesh.ygrid, 2] = mesh.u_left_boundary.value
        mesh.a_pressure_boundary[0:mesh.ygrid] = 0
    if mesh.u_right_boundary == 'D':
        mesh.vel_face[mesh.num_nodes - mesh.ygrid:, 0] = mesh.u_right_boundary.value
        mesh.u_vel_boundaries[mesh.num_nodes - mesh.ygrid:, 0] = mesh.u_right_boundary.value
        mesh.a_pressure_boundary[mesh.num_nodes - mesh.ygrid:] = 0
    if mesh.u_bottom_boundary.type == 'D':
        mesh.u_vel_boundaries[0:-1:mesh.ygrid, 3] = mesh.u_bottom_boundary.value
    if mesh.u_top_boundary.type == 'D':
        mesh.u_vel_boundaries[mesh.ygrid - 1:-1:mesh.ygrid, 1] = mesh.u_top_boundary.value

    # V boundaries
    if mesh.v_left_boundary.type == 'D':
        mesh.v_vel_boundaries[0:mesh.ygrid, 0] = mesh.v_left_boundary.value
    if mesh.v_right_boundary.type == 'D':
        mesh.v_vel_boundaries[mesh.num_nodes - mesh.ygrid:, 0] = mesh.v_right_boundary.value
    if mesh.v_bottom_boundary == 'D':
        mesh.vel_face[0:-1:mesh.ygrid, 3] = mesh.v_bottom_boundary.value
        mesh.v_vel_boundaries[0:-1:mesh.ygrid, 3] = mesh.v_bottom_boundary.value
        mesh.a_pressure_boundary[0:-1:mesh.ygrid] = 0
    if mesh.v_top_boundary == 'D':
        mesh.vel_face[mesh.ygrid - 1:-1:mesh.ygrid, 1] = mesh.v_top_boundary.value
        mesh.v_vel_boundaries[mesh.ygrid - 1:-1:mesh.ygrid, 1] = mesh.v_top_boundary.value
        mesh.a_pressure_boundary[mesh.ygrid - 1:-1:mesh.ygrid] = 0


@jit(nopython=True, parallel=True)
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
        mesh.a_momentum[idx + mesh.ygrid, 3] = mesh.a_momentum[idx, 1]
        mesh.a_momentum[idx + 1, 4] = mesh.a_momentum[idx, 2]
        mesh.momentum_pressure_source[idx, 0] = - dpx_p * mesh.volumes[idx]  # TODO: verify sign here and below
        mesh.momentum_pressure_source[idx, 1] = - dpy_p * mesh.volumes[idx]
    return mesh.a_momentum


def momentum_solver(mesh):  # TODO: this function
    pass


@jit(nopython=True, parallel=True)
def face_velocities(mesh):
    # Nodes not on boundaries
    for idx in range(mesh.num_nodes - mesh.ygrid):
        dpx_E, dpx_p, dpx_e, dpy_N, dpy_p, dpy_n = pressure_derivatives(mesh, idx)
        if idx >= mesh.ygrid * mesh.xgrid - mesh.ygrid:  # Right boundary adjacent
            continue
        elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary adjacent
            continue
        else:
            mesh.vel_face[idx, 0] = (mesh.velocities[idx, 0] + mesh.velocities[idx + mesh.ygrid, 0]) / 2 \
                                    + (1 / 2) * (mesh.volumes[idx] / mesh.a_momentum[idx, 0] * dpx_p
                                                 + mesh.volumes[idx + mesh.ygrid] /
                                                 mesh.a_momentum[idx + mesh.ygrid, 0] * dpx_E) - \
                                    (mesh.volumes[idx] / 2 + mesh.volumes[idx + mesh.ygrid] / 2) * \
                                    (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                                        idx + mesh.ygrid, 0]) \
                                    * dpx_e
            mesh.vel_face[idx, 1] = (mesh.velocities[idx, 1] + mesh.velocities[idx + mesh.ygrid, 1]) / 2 \
                                    + (1 / 2) * (mesh.volumes[idx] / mesh.a_momentum[idx, 0] * dpy_p
                                                 + mesh.volumes[idx + mesh.ygrid] /
                                                 mesh.a_momentum[idx + mesh.ygrid, 0] * dpy_N) - \
                                    (mesh.volumes[idx] / 2 + mesh.volumes[idx + mesh.ygrid] / 2) * \
                                    (1 / mesh.a_momentum[idx + mesh.ygrid, 0] + 1 / mesh.a_momentum[
                                        idx + mesh.ygrid, 0]) \
                                    * dpy_n
        # Assign w and s face velocities  # TODO: not sure if needed or not?
        mesh.vel_face[idx + mesh.ygrid, 2] = mesh.vel_face[idx, 0]
        mesh.vel_face[idx + 1, 3] = mesh.vel_face[idx, 1]


@jit()
def pressure_derivatives(mesh, idx):  # TODO: pressure_boundary variable and check top calculations
    if idx >= mesh.num_nodes - mesh.ygrid:  # Right boundary
        if idx == mesh.num_nodes - 1:  # Top right corner
            dpx_p = (2 * mesh.pressure_boundary[idx - mesh.ygrid * mesh.column[idx] + 1, 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundary[mesh.column[idx] + 1, 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        elif idx == mesh.num_nodes - mesh.ygrid:  # Bottom right corner
            dpx_p = (2 * mesh.pressure_boundary[idx - mesh.ygrid * mesh.column[idx] + 1, 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundary[mesh.column[idx] + 1, 3] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        else:
            dpx_p = (2 * mesh.pressure_boundary[idx - mesh.ygrid * mesh.column[idx] + 1, 2] -
                     mesh.pressure[idx - mesh.ygrid] -
                     mesh.pressure[idx]) / (2 * mesh.areas[idx, 1])
            dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        return _, dpx_p, _, _, dpy_p, _
    elif idx % mesh.ygrid == (mesh.ygrid - 1):  # Top boundary
        if idx == mesh.num_nodes - 1:  # Top left corner
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2 * mesh.pressure_boundary[idx + 1, 0]) \
                    / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundary[mesh.column[idx] + 1, 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        else:
            dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
            dpy_p = (2 * mesh.pressure_boundary[mesh.column[idx] + 1, 1] - mesh.pressure[idx]
                     - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        return _, dpx_p, _, _, dpy_p, _
    else:  # Added interpolations for face velocity calculations
        dpx_E = (mesh.pressure[idx + mesh.ygrid * 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 1])
        dpx_p = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx - mesh.ygrid]) / (2 * mesh.areas[idx, 1])
        dpx_e = (mesh.pressure[idx + mesh.ygrid] - mesh.pressure[idx]) / (2 * mesh.dx[mesh.column + 1])
        dpy_N = (mesh.pressure[idx + 2] - mesh.pressure[idx]) / (2 * mesh.areas[idx + mesh.ygrid, 0])
        dpy_p = (mesh.pressure[idx + 1] - mesh.pressure[idx - 1]) / (2 * mesh.areas[idx, 0])
        dpy_n = (mesh.pressure[idx + 1] - mesh.pressure[idx]) / (2 * mesh.dy[idx + 1])
        if idx == 0:  # Bottom left boundary corner adjacent
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2 * mesh.pressure_boundary[idx + 1, 0]) \
                    / (2 * mesh.areas[idx, 1])
            dpy_p = (mesh.pressure[idx + 1] + mesh.pressure[idx] - 2 * mesh.pressure_boundary[idx + 1, 3]) / (
                    2 * mesh.areas[idx, 0])
        elif idx < mesh.ygrid:  # Left boundary adjacent
            dpx_p = (mesh.pressure[idx + mesh.ygrid] + mesh.pressure[idx] - 2 * mesh.pressure_boundary[idx + 1, 0]) \
                    / (2 * mesh.areas[idx, 1])
        elif idx % mesh.ygrid == 0 and idx != 0:  # Bottom boundary adjacent
            dpy_p = (mesh.pressure[idx + 1] + mesh.pressure[idx] -
                     2 * mesh.pressure_boundary[mesh.column[idx] + 1, 3]) / (2 * mesh.areas[idx, 0])
        elif idx > mesh.num_nodes - 2 * mesh.ygrid & idx < mesh.num_nodes - mesh.ygrid:  # Two from right boundary
            dpx_E = (2 * mesh.pressure_boundary[idx % mesh.ygrid + 1, 2] -
                     mesh.pressure[idx] - mesh.pressure[idx + mesh.ygrid]) \
                    / (2 * mesh.areas[idx + mesh.ygrid, 1])
        elif idx % mesh.ygrid == (mesh.ygrid - 2):  # Two from top boundary
            dpy_N = (2 * mesh.pressure_boundary[mesh.column[idx] + 1, 1] - mesh.pressure[idx] - mesh.pressure[
                idx + 1]) / \
                    (2 * mesh.areas[idx + mesh.ygrid, 0])
        return dpx_E, dpx_p, dpx_e, dpy_N, dpy_p, dpy_n


def pressure_correction_formulation(mesh):  # TODO: this function
    f_e = np.zeros(mesh.num_nodes)
    f_w = np.zeros(mesh.num_nodes)
    f_n = np.zeros(mesh.num_nodes)
    f_s = np.zeros(mesh.num_nodes)
    for idx in range(mesh.num_nodes):
        f_e[idx] = mesh.rhos * mesh.vel_face[idx, 0] * mesh.areas[idx, 0]
        f_n[idx] = mesh.rhos * mesh.vel_face[idx, 1] * mesh.areas[idx, 1]
        f_w[idx] = mesh.rhos * mesh.vel_face[idx, 2] * mesh.areas[idx, 0]
        f_s[idx] = mesh.rhos * mesh.vel_face[idx, 3] * mesh.areas[idx, 1]
        if mesh.a_pressure_boundary[idx] == 1:
            mesh.a_pressure[idx, 1] = (mesh.rhos * mesh.areas[idx, 0] ** 2 / 2) * (
                        1 / mesh.a_momentum[idx + mesh.ygrid, 0]
                        + 1 / mesh.a_momentum[idx, 0])
            mesh.a_pressure[idx, 2] = (mesh.rhos * mesh.areas[idx, 1] ** 2 / 2) * (1 / mesh.a_momentum[idx + 1, 0]
                                                                                   + 1 / mesh.a_momentum[idx, 0])
            mesh.a_pressure[idx, 3] = (mesh.rhos * mesh.areas[idx, 0] ** 2 / 2) * (
                        1 / mesh.a_momentum[idx - mesh.ygrid, 0]
                        + 1 / mesh.a_momentum[idx, 0])
            mesh.a_pressure[idx, 4] = (mesh.rhos * mesh.areas[idx, 1] ** 2 / 2) * (1 / mesh.a_momentum[idx - 1, 0]
                                                                                   + 1 / mesh.a_momentum[idx, 0])
        else:
            mesh.a_pressure[idx, 1] = 0
            mesh.a_pressure[idx, 2] = 0
            mesh.a_pressure[idx, 3] = 0
            mesh.a_pressure[idx, 4] = 0
        mesh.pressure_source[idx] = f_w[idx] - f_e[idx] + f_s[idx] - f_n[idx]
        mesh.a_pressure[idx, 0] = mesh.a_pressure[idx, 1] + mesh.a_pressure[idx, 2] + mesh.a_pressure[idx, 3] + \
                                  mesh.a_pressure[idx, 4]


def pressure_correction_solver(mesh):  # TODO: this function
    pass


def correct_nodal_velocities(mesh):  # TODO: this function
    pass


def correct_pressure(mesh):  # TODO: this function
    pass


def correct_face_velocities(mesh):  # TODO: this function
    pass


def pressure_extrapolation(mesh):  # TODO: this function
    pass


# Save mesh data (pressure and velocity field)
def save_mesh_data(mesh):
    pass


# Node functions

# Discretization Functions

# Solver Functions

# Visualization Functions
def fvm_solver(mesh):
    set_boundary_values(mesh)
    momentum_formulation(mesh)
    momentum_solver(mesh)
    face_velocities(mesh)
    pressure_correction_formulation(mesh)
    pressure_correction_solver(mesh)
    correct_nodal_velocities(mesh)
    correct_pressure(mesh)
    correct_face_velocities(mesh)
    pressure_extrapolation(mesh)


def main():
    # LID CAVITY FLOW PROBLEM
    # Constants and input parameters
    u_top = 1
    p_top = 1
    object = False
    # Create boundaries
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

    # Create Mesh
    mesh1 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 10, 10, 1, 1)
    # fvm_solver(mesh1)
    set_boundary_values(mesh1)
    # momentum_formulation(mesh1)
    # momentum_solver(mesh1)
    # face_velocities(mesh1)
    # pressure_correction_formulation(mesh1)
    # pressure_correction_solver(mesh1)
    # correct_nodal_velocities(mesh1)
    # correct_pressure(mesh1)
    # correct_face_velocities(mesh1)
    # pressure_extrapolation(mesh1)

    start = time.time()
    # mesh1.create_nodes()
    end = time.time()
    print("Elapsed (without compilation) = %s" % (end - start))
    # start = time.time()
    # mesh2 = Mesh('2D', boundaries1)
    # mesh2.generate_2d_uniform_mesh(320, 320, 1, 1)
    # end = time.time()
    # print("Elapsed (with compilation) = %s" % (end - start))

    # LID DRIVEN CAVITY WITH STEP PROBLEM
    object = True

    # BACK-STEP FLOW PROBLEM
    object = True


if __name__ == '__main__':
    main()
