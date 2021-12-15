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
        self.vel = np.zeros(((x_points+2)*(y_points+2), 2))
        self.vel_correction = np.zeros(((x_points + 2) * (y_points + 2), 2))
        self.pressure = np.zeros(((x_points+2)*(y_points+2), 1))
        self.pressure_correction = np.zeros(((x_points + 2) * (y_points + 2), 1))
        self.vel_face = np.zeros(((x_points + 2) * (y_points + 2), 2))
        self.vel_face_correction = np.zeros(((x_points + 2) * (y_points + 2), 2))
        self.a_momentum = np.zeros(((x_points + 2) * (y_points + 2), 5))
        self.momentum_source = np.zeros(((x_points + 2) * (y_points + 2), 2))
        self.a_pressure = np.zeros(((x_points + 2) * (y_points + 2), 5))
        self.pressure_source = np.zeros(((x_points + 2) * (y_points + 2), 2))

    def cell_volumes_areas_deltas(self):
        self.volumes = np.zeros((self.num_nodes, 1))
        self.areas = np.zeros((self.num_nodes, 2))
        for i in range(self.num_nodes):
            if i == 0:  # Bottom left boundary corner
                self.areas[i, 0] = self.dy[0] + self.dy[1] / 2  # Ay
                self.areas[i, 1] = self.dx[0] + self.dx[1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i == self.ygrid-1:  # Top left corner boundary
                self.areas[i, 0] = self.dy[-1] + self.dy[-2] / 2  # Ay
                self.areas[i, 1] = self.dx[0] + self.dx[1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i == self.ygrid * self.xgrid - self.ygrid:  # Bottom right corner boundary
                self.areas[i, 0] = self.dy[0] + self.dy[1] / 2  # Ay
                self.areas[i, 1] = self.dx[-1] + self.dx[-2] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i == self.ygrid * self.xgrid - 1:  # Top right corner boundary
                self.areas[i, 0] = self.dy[-1] + self.dy[-2] / 2  # Ay
                self.areas[i, 1] = self.dx[-1] + self.dx[-2] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i < self.ygrid:  # Left boundary
               self.areas[i, 0] = self.dy[i + 1] / 2 + self.dy[i] / 2  # Ay
               self.areas[i, 1] = self.dx[0] + self.dx[1] / 2  # Ax
               self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i >= self.ygrid * self.xgrid - self.ygrid:  # Right boundary
                self.areas[i, 0] = self.dy[(i - self.column * self.ygrid) + 1] / 2 + \
                                   self.dy[i - self.column * self.ygrid] / 2  # Ay
                self.areas[i, 1] = self.dx[-1] + self.dx[-2] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i % self.ygrid == 0 and i != 0:  # Bottom boundary
                self.areas[i, 0] = self.dy[0] + self.dy[1] / 2  # Ay
                self.areas[i, 1] = self.dx[self.column] / 2 + self.dx[self.column + 1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            elif i % self.ygrid == (self.ygrid - 1):  # Top boundary
                self.areas[i, 0] = self.dy[-1] + self.dy[-2] / 2  # Ay
                self.areas[i, 1] = self.dx[self.column] / 2 + self.dx[self.column + 1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]
            else:
                self.areas[i, 0] = self.dy[(i - self.column * self.ygrid) + 1] / 2 + \
                                   self.dy[i - self.column * self.ygrid] / 2 # Ay
                self.areas[i, 1] = self.dx[self.column] / 2 + self.dx[self.column + 1] / 2  # Ax
                self.volumes[i] = self.areas[i, 0] * self.areas[i, 1]

    def set_rho(self, rhos):
        self.rhos = rhos

    def set_gamma(self, gammas):
        self.gammas = gammas

    # Node interaction functions
    def create_nodes(self):
        self.nodes = np.empty(self.num_nodes, dtype=Node)
        self.nodes = [Node() for i in range(self.num_nodes)]

    def assign_coordinates_to_nodes(self):
        [(self.nodes[idx].set_column(idx, self), self.nodes[idx].set_coordinates(idx, self)) for idx in
         range(self.num_nodes)]

    def assign_rho_to_nodes(self, rhos):
        [node.set_rho(rhos[i]) for i, node in enumerate(self.nodes)]


class Node:
    def __init__(self):
        self.column = None
        self.x_coordinate = None
        self.y_coordinate = None

    def set_rho(self, rho):
        self.rho = rho

    def set_column(self, idx, mesh):
        self.column = int(np.floor(idx / mesh.ygrid))

    def set_coordinates(self, index, mesh):
        self.x_coordinate = np.sum(mesh.dx[:1 + self.column])
        self.y_coordinate = np.sum(mesh.dy[:1 + index - self.column * mesh.ygrid])

    def set_node_neighbors(self, mesh):
        pass

    def set_node_types(self, boundaries):
        self.codes = np.zeros(self.num_nodes)
        self.boundary_values = np.zeros(self.num_nodes)
        for boundary in boundaries:
            self.boundary_type(boundary)

    def boundary_type(self, boundary):
        if boundary.type == 'D':
            if boundary.location.lower() == 'left':
                self.codes[:self.size[1]] += 1
            elif boundary.location.lower() == 'right':
                self.codes[self.num_nodes - self.size[1]:] += 1
            elif boundary.location.lower() == 'top':
                self.codes[np.arange(self.size[1] - 1, self.num_nodes, self.size[1])] += 1
            elif boundary.location.lower() == 'bottom':
                self.codes[np.arange(self.size[1], self.num_nodes, self.size[1])] += 1
            else:
                TypeError('Incorrect location entered')


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
# Save mesh data (pressure and velocity field)
def save_mesh_data(mesh):
    pass


def face_velocities(mesh):
    # Nodes not on boundaries
    for i in range(mesh.num_nodes):
        pressures = pressure_interpolation(mesh, i)
        # Interior nodes
        if i > mesh.ygrid:
            # East face velocity
            mesh.vel_face[i, 0] = (mesh.velocities[i] + mesh.velocites[i + mesh.ygrid]) / 2 \
                                  + (1 / 2)\
                                  * ((mesh.volumes[i]/mesh.a_momentum[i]) *
                                     ((mesh.pressure[i + mesh.ygrid*2]-mesh.pressure[i])
                                      / (2*mesh.areas[i + mesh.ygrid, 1])) +
                                     (mesh.volumes[i+mesh.ygrid] / mesh.a_momentum[i + mesh.ygrid]) *
                                     ((mesh.pressure[i + mesh.ygrid] - mesh.pressure[i - mesh.ygrid]) /
                                      (2 * mesh.areas[i, 1]))) \
                                  - (mesh.volumes[i] / 2 + mesh.volumes[i+mesh.ygrid] / 2) * \
                                  (1 / mesh.a_momentum[i + mesh.ygrid] + 1 / mesh.a_momentum[i + mesh.ygrid]) * \
                                  ((mesh.pressure[i+mesh.ygrid] - mesh.pressure[i]) /
                                   (2*mesh.dx[mesh.column + 1]))

            # North face velocity
            mesh.vel_face[i, 1] = (mesh.velocities[i] + mesh.velocites[i + mesh.ygrid]) / 2 + (1 / 2)
        # Nodes next to left boundary

        # Nodes two from right boundary

        # Nodes two from top boundary


def pressure_interpolation(mesh, index):  # TODO
    pass

def set_boundary_values(mesh):
    pass
# Node functions

# Discretization Functions

# Solver Functions

# Visualization Functions


def main():
    # LID CAVITY FLOW PROBLEM
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
    boundary_right_p = Boundary('D', 0, 'right')
    boundary_bottom_p = Boundary('D', 0, 'bottom')
    # Boundary set for domain
    boundaries_u = [boundary_left_u, boundary_right_u, boundary_top_u, boundary_bottom_u]
    boundaries_v = [boundary_left_v, boundary_right_v, boundary_top_v, boundary_bottom_v]
    boundaries_p = [boundary_left_p, boundary_right_p, boundary_top_p, boundary_bottom_p]

    # Create Mesh
    mesh1 = Mesh('2D_uniform', boundaries_u, boundaries_v, boundaries_p, 320, 320, 1, 1)
    start = time.time()
    mesh1.generate_2d_uniform_mesh(320, 320, 1, 1)
    mesh1.create_nodes()
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
