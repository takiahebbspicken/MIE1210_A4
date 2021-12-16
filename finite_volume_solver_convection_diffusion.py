# MIE1210 - Assignment #3 - Finite-Volume Solver
# Created by: Takiah Ebbs-Picken, takiah.ebbspicken@mail.utoronto.ca
# Modified November 2021

import time
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# User input:
solver_tolerance = 1e-6  # Iterative solver tolerance
solver_max_iterations = 1000  # Iterative solver maximum number of iterations
inflation_factor = 1
length_x = 1  # Length of x domain in [m]
length_y = 1  # Length of y domain in [m]
grid_points_x = 80  # Number of nodes in the x direction
grid_points_y = 80  # Number of nodes in the y direction
# Specify boundary types and values
boundary_left_type = 'D'
boundary_left_value = 100
boundary_right_type = 'D'
boundary_right_value = 0
boundary_top_type = 'D'
boundary_top_value = 100
boundary_bottom_type = 'D'
boundary_bottom_value = 0
discretization_type = 'UW'  # Specify discretization type: 'CD' central differencing scheme, or 'UW' upwind scheme
velocity_field = True  # Turn velocity field on or off, if off: u_x and u_y specified below, if on: circular field
# Specify constants
gamma = 5
u_x = 2
u_y = 2
# Turn data saving on or off
save_data = True


def delta(grid, length, r=1):
    # Regular deltas for mesh with no inflation
    if r == 1:
        d = length / grid
        deltas = np.ones((grid + 1, 1)) * d
        deltas[0] = d / 2
        deltas[-1] = d / 2
    # Deltas when there is inflation
    elif r > 1:
        deltas = np.zeros((grid + 1, 1))
        deltas[0] = ((1 - r) / (1 - r ** (grid / 2))) * (length / 2)
        for k in range(1, grid + 1):
            coord = np.sum(deltas)
            if coord <= (length / 2):
                deltas[k] = r * deltas[k - 1]
            else:
                deltas[k] = deltas[k - 1] / r
    # Scale deltas to ensure mesh has not been extended past the actual domain
    if np.sum(deltas) > length:
        rescale_factor = length / np.sum(deltas)
        deltas = deltas * rescale_factor
    return deltas


def two_dim_rectangular_mesh_delta(grid_x, grid_y, l_x, l_y, r=1):
    # Generate mesh coordinates with nodes on boundaries for testing purposes and mesh visualization
    num_nodes = (grid_x + 2) * (grid_y + 2)
    dx = delta(grid_x, l_x, r)
    dy = delta(grid_y, l_y, r)
    node_x_coordinates = np.zeros((grid_x + 2))
    node_y_coordinates = np.zeros((grid_y + 2))
    for i in range(1, grid_x + 2):
        node_x_coordinates[i] = node_x_coordinates[i - 1] + dx[i - 1]
    for i in range(1, grid_y + 2):
        node_y_coordinates[i] = node_y_coordinates[i - 1] + dy[i - 1]
    nodes = np.zeros((num_nodes, 2))
    for i in range(len(node_y_coordinates)):
        y_coords = node_y_coordinates[i] * np.ones(len(node_x_coordinates))
        nodes[i * (grid_x + 2):i * (grid_x + 2) + (grid_x + 2), 0] = node_x_coordinates
        nodes[i * (grid_x + 2):i * (grid_x + 2) + (grid_x + 2), 1] = y_coords
    return nodes


def two_dim_rectangular_mesh_no_boundary(grid_x, grid_y, l_x, l_y, r=1):
    # Generate node point coordinates for plotting purposes
    num_nodes = grid_x * grid_y
    dx = delta(grid_x, l_x, r)
    dy = delta(grid_y, l_y, r)
    node_x_coordinates = np.zeros((grid_x))
    node_y_coordinates = np.zeros((grid_y))
    node_x_coordinates[0] = dx[0]
    node_y_coordinates[0] = dy[0]
    for i in range(1, grid_x):
        node_x_coordinates[i] = node_x_coordinates[i - 1] + dx[i]
    for i in range(1, grid_y):
        node_y_coordinates[i] = node_y_coordinates[i - 1] + dy[i]
    nodes = np.zeros((num_nodes, 2))
    for i in range(len(node_x_coordinates)):
        x_coords = node_x_coordinates[i] * np.ones(len(node_y_coordinates))
        nodes[i * (grid_y):i * (grid_y) + (grid_y), 1] = node_y_coordinates
        nodes[i * (grid_y):i * (grid_y) + (grid_y), 0] = x_coords
    return nodes, node_x_coordinates, node_y_coordinates


# Following functions are the equations for various discretizations depending on boundary conditions


def radius_coordinate(l_x, l_y, x_coordinate, y_coordinate):
    return np.sqrt((x_coordinate - l_x / 2) ** 2 + (y_coordinate - l_y / 2) ** 2)


def theta_coordinate(l_x, l_y, x_coordinate, y_coordinate):
    return np.arctan2(y_coordinate - l_y / 2, x_coordinate - l_x / 2)


def vel_field_x(r, theta):
    return -r * np.sin(theta)


def vel_field_y(r, theta):
    return r * np.cos(theta)


def diffusion_coefficient(gamma, area, delta_x_y):
    # Function for D_i
    return (gamma * area) / delta_x_y


def convection_coefficient(velocity, area, density=1):
    # Function for F_i
    return velocity * area * density


def ap(a_1, a_2, a_3, a_4, f_1, f_2, f_3, f_4, s_p=0):
    # Function for computing interior node a_p
    return a_1 + a_2 + a_3 + a_4 - s_p + (f_1 - f_2 + f_3 - f_4)


def a_en_central_difference(d, f):
    # Interior node a_n, a_e equation
    return d - f / 2


def a_ws_central_difference(d, f):
    # Interior node a_n, a_e equation
    return d + f / 2


def a_en_upwind(d, f):
    # Interior node a_n, a_e equation
    if -f > 0:
        return d - f
    else:
        return d


def a_ws_upwind(d, f):
    # Interior node a_s, a_w equation
    if f > 0:
        return d + f
    else:
        return d


def interior_source(s_u=0):
    # Interior node source
    return s_u


def dirichlet_boundary_source(a_b, phi_b, s_u=0):
    # Equation for source term at dirichlet boundary
    return a_b * phi_b + s_u


def dirichlet_dirichlet_boundary_source(a_b1, a_b2, phi_b1, phi_b2, s_u=0):
    # Equation for source term at dirichlet boundary
    return a_b1 * phi_b1 + s_u + a_b2 * phi_b2


# Initiate b matrices and diagonals
b_matrix = np.zeros((grid_points_y * grid_points_x, 1))
super_diag = np.zeros((grid_points_y * grid_points_x))
diag = np.zeros((grid_points_y * grid_points_x))
sub_diag = np.zeros((grid_points_y * grid_points_x))
upper_off_diag = np.zeros((grid_points_y * grid_points_x))
lower_off_diag = np.zeros((grid_points_y * grid_points_x))

# Compute dxs and dys
dxs = delta(grid_points_x, length_x, inflation_factor)
dys = delta(grid_points_y, length_y, inflation_factor)

column_counter = 0
aps = np.zeros((grid_points_y * grid_points_x, 1))
t0 = time.time()
for i in range(grid_points_y * grid_points_x):
    if i % grid_points_y == 0 and i != 0:
        column_counter += 1
    if i == 0:  # Bottom left corner boundary
        if velocity_field:
            ux_e = vel_field_x(radius_coordinate(length_x, length_y, dxs[0] + dxs[1] / 2, dys[0]),
                               theta_coordinate(length_x, length_y, dxs[0] + dxs[1] / 2, dys[0]))
            uy_n = vel_field_y(radius_coordinate(length_x, length_y, dxs[0], dys[0] + dys[1] / 2),
                               theta_coordinate(length_x, length_y, dxs[0], dys[0] + dys[1] / 2))
            uy_b_bottom = vel_field_y(radius_coordinate(length_x, length_y, dxs[0], 0),
                                      theta_coordinate(length_x, length_y, dxs[0], 0))
            ux_b_left = vel_field_x(radius_coordinate(length_x, length_y, 0, dys[0]),
                                    theta_coordinate(length_x, length_y, 0, dys[0]))
        else:
            ux_e = u_x
            uy_n = u_y
            uy_b_bottom = u_y
            ux_b_left = u_x
        if discretization_type == 'CD':
            a_n = a_en_central_difference(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[1]),
                                          convection_coefficient(uy_n, dxs[0] + dxs[1] / 2))
            a_e = a_en_central_difference(diffusion_coefficient(gamma, dys[0] + dys[1] / 2, dxs[1]),
                                          convection_coefficient(ux_e, dys[0] + dys[1] / 2))
        else:
            a_n = a_en_upwind(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[1]),
                              convection_coefficient(uy_n, dxs[0] + dxs[1] / 2))
            a_e = a_en_upwind(diffusion_coefficient(gamma, dys[0] + dys[1] / 2, dxs[1]),
                              convection_coefficient(ux_e, dys[0] + dys[1] / 2))
        if boundary_left_type == 'D':
            if boundary_bottom_type == 'D':
                if discretization_type == 'CD':
                    a_b_bottom = a_ws_central_difference(diffusion_coefficient(gamma,
                                                                               dys[0] + dys[1] / 2, dxs[0]),
                                                         convection_coefficient(uy_b_bottom, dys[0] + dys[1]
                                                                                / 2))
                    a_b_left = a_ws_central_difference(diffusion_coefficient(gamma,
                                                                             dxs[0] + dxs[1] / 2, dxs[0]),
                                                       convection_coefficient(ux_b_left, dxs[0] + dxs[1] / 2))
                else:
                    a_b_bottom = a_ws_upwind(diffusion_coefficient(gamma, dys[0] + dys[1] / 2, dxs[0]),
                                             convection_coefficient(uy_b_bottom, dys[0] + dys[1] / 2))
                    a_b_left = a_ws_upwind(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dxs[0]),
                                           convection_coefficient(ux_b_left, dxs[0] + dxs[1] / 2))
                aps[i] = ap(a_e, a_b_left, a_n, a_b_bottom, convection_coefficient(ux_e, dys[0] + dys[1] / 2),
                            convection_coefficient(ux_b_left, dxs[0] + dxs[1] / 2),
                            convection_coefficient(uy_n, dxs[0] + dxs[1] / 2),
                            convection_coefficient(uy_b_bottom, dys[0] + dys[1] / 2))
                b_matrix[i] = dirichlet_dirichlet_boundary_source(a_b_bottom, a_b_left, boundary_bottom_value,
                                                                  boundary_left_value)

    elif i == grid_points_y - 1:  # Top left corner boundary
        if velocity_field:
            ux_e = vel_field_x(radius_coordinate(length_x, length_y, dxs[0] + dxs[1] / 2, length_y - dys[-1]),
                               theta_coordinate(length_x, length_y, dxs[0] + dxs[1] / 2, length_y - dys[-1]))
            uy_s = vel_field_y(radius_coordinate(length_x, length_y, dxs[0], length_y - dys[-1] - dys[-2] / 2),
                               theta_coordinate(length_x, length_y, dxs[0], length_y - dys[-1] - dys[-2] / 2))
            uy_b_top = vel_field_y(radius_coordinate(length_x, length_y, dxs[0], length_y),
                                   theta_coordinate(length_x, length_y, dxs[0], length_y))
            ux_b_left = -vel_field_x(radius_coordinate(length_x, length_y, 0, dys[0]),
                                     theta_coordinate(length_x, length_y, 0, dys[0]))
        else:
            ux_e = u_x
            uy_s = u_y
            uy_b_top = u_y
            ux_b_left = u_x
        if discretization_type == 'CD':
            a_e = a_en_central_difference(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2, dxs[1]),
                                          convection_coefficient(ux_e, dys[-1] + dys[-2] / 2))
            a_s = a_ws_central_difference(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[-2]),
                                          convection_coefficient(uy_s, dxs[0] + dxs[1] / 2))
        else:
            a_e = a_en_upwind(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2, dxs[1]),
                              convection_coefficient(ux_e, dys[-1] + dys[-2] / 2))
            a_s = a_ws_upwind(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[-2]),
                              convection_coefficient(uy_s, dxs[0] + dxs[1] / 2))
        if boundary_left_type == 'D':
            if boundary_top_type == 'D':
                if discretization_type == 'CD':
                    a_b_top = a_en_central_difference(diffusion_coefficient(gamma,
                                                                            dxs[0] + dxs[1] / 2, dys[-1]),
                                                      convection_coefficient(uy_b_top, dxs[0] + dxs[1] / 2))
                    a_b_left = a_ws_central_difference(diffusion_coefficient(gamma,
                                                                             dys[-1] + dys[-2] / 2, dxs[0]),
                                                       convection_coefficient(ux_b_left,
                                                                              dys[-1] + dys[-2] / 2))
                else:
                    a_b_top = a_en_upwind(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[-1]),
                                          convection_coefficient(uy_b_top, dxs[0] + dxs[1] / 2))
                    a_b_left = a_ws_upwind(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2, dxs[0]),
                                           convection_coefficient(ux_b_left, dys[-1] + dys[-2] / 2))
                aps[i] = ap(a_e, a_b_left, a_b_top, a_s, convection_coefficient(ux_e, dys[-1] + dys[-2] / 2),
                            convection_coefficient(ux_b_left, dys[-1] + dys[-2] / 2),
                            convection_coefficient(uy_b_top, dxs[0] + dxs[1] / 2),
                            convection_coefficient(uy_s, dxs[0] + dxs[1] / 2))
                b_matrix[i] = dirichlet_dirichlet_boundary_source(a_b_left, a_b_top, boundary_left_value,
                                                                  boundary_top_value)

    elif i == grid_points_y * grid_points_x - grid_points_y:  # Bottom right corner boundary
        if velocity_field:
            uy_n = vel_field_y(radius_coordinate(length_x, length_y, length_x - dxs[-1],
                                                 dys[0] + dys[1] / 2),
                               theta_coordinate(length_x, length_y, length_x - dxs[-1],
                                                dys[0] + dys[1] / 2))
            ux_w = vel_field_x(radius_coordinate(length_x, length_y, length_x - dxs[-1] - dxs[-2] / 2, dys[0]),
                               theta_coordinate(length_x, length_y, length_x - dxs[-1] - dxs[-2] / 2, dys[0]))
            uy_b_bottom = vel_field_y(radius_coordinate(length_x, length_y, length_x - dxs[-1], 0),
                                      theta_coordinate(length_x, length_y, length_x - dxs[-1], 0))
            ux_b_right = vel_field_x(radius_coordinate(length_x, length_y, length_x, dys[0]),
                                     theta_coordinate(length_x, length_y, length_x, dys[0]))
        else:
            uy_n = u_y
            ux_w = u_x
            uy_b_bottom = u_y
            ux_b_right = u_x
        if discretization_type == 'CD':
            a_n = a_en_central_difference(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2, dys[1]),
                                          convection_coefficient(uy_n, dxs[-1] + dxs[-2] / 2))
            a_w = a_ws_central_difference(diffusion_coefficient(gamma, dys[0] + dys[1] / 2, dxs[-2]),
                                          convection_coefficient(ux_w, dys[0] + dys[1] / 2))
        else:
            a_n = a_en_upwind(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2, dys[1]),
                              convection_coefficient(uy_n, dxs[-1] + dxs[-2] / 2))
            a_w = a_ws_upwind(diffusion_coefficient(gamma, dys[0] + dys[1] / 2, dxs[-2]),
                              convection_coefficient(ux_w, dys[0] + dys[1] / 2))
        if boundary_right_type == 'D':
            if boundary_bottom_type == 'D':
                if discretization_type == 'CD':
                    a_b_bottom = a_ws_central_difference(diffusion_coefficient(gamma,
                                                                               dxs[-1] + dxs[-2] / 2, dys[0]),
                                                         convection_coefficient(uy_b_bottom, dxs[-1] + dxs[-2]
                                                                                / 2))
                    a_b_right = a_en_central_difference(diffusion_coefficient(gamma,
                                                                              dys[0] + dys[1] / 2, dxs[-1]),
                                                        convection_coefficient(ux_b_right,
                                                                               dys[0] + dys[1] / 2))
                else:
                    a_b_bottom = a_ws_upwind(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2, dys[0]),
                                             convection_coefficient(uy_b_bottom, dxs[-1] + dxs[-2] / 2))
                    a_b_right = a_en_upwind(diffusion_coefficient(gamma, dys[0] + dys[1] / 2, dxs[-1]),
                                            convection_coefficient(ux_b_right, dys[0] + dys[1] / 2))
                aps[i] = ap(a_b_right, a_w, a_n, a_b_bottom, convection_coefficient(ux_b_right, dys[0] + dys[1] / 2),
                            convection_coefficient(ux_w, dys[0] + dys[1] / 2),
                            convection_coefficient(uy_n, dxs[-1] + dxs[-2] / 2),
                            convection_coefficient(uy_b_bottom, dxs[-1] + dxs[-2] / 2))
                b_matrix[i] = dirichlet_dirichlet_boundary_source(a_b_bottom, a_b_right, boundary_bottom_value,
                                                                  boundary_right_value)

    elif i == grid_points_y * grid_points_x - 1:  # Top right corner boundary
        if velocity_field:
            uy_s = vel_field_y(radius_coordinate(length_x, length_y, length_x - dxs[-1],
                                                 length_y - dys[-1] - dys[-2] / 2),
                               theta_coordinate(length_x, length_y, length_x - dxs[-1],
                                                length_y - dys[-1] - dys[-2] / 2))
            ux_w = vel_field_x(
                radius_coordinate(length_x, length_y, length_x - dxs[-1] - dxs[-2] / 2, length_y - dys[-1]),
                theta_coordinate(length_x, length_y, length_x - dxs[-1] - dxs[-2] / 2, length_y - dys[-1]))
            uy_b_top = vel_field_y(radius_coordinate(length_x, length_y, length_x - dxs[-1], length_y),
                                   theta_coordinate(length_x, length_y, length_x - dxs[-1], length_y))
            ux_b_right = vel_field_x(radius_coordinate(length_x, length_y, length_x, length_y - dys[-1]),
                                     theta_coordinate(length_x, length_y, length_x, length_y - dys[-1]))
        else:
            uy_s = u_y
            ux_w = u_x
            uy_b_top = u_y
            ux_b_right = u_x
        if discretization_type == 'CD':
            a_w = a_ws_central_difference(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2, dxs[-2]),
                                          convection_coefficient(ux_w, dys[-1] + dys[-2] / 2))
            a_s = a_ws_central_difference(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2, dys[-2]),
                                          convection_coefficient(uy_s, dxs[-1] + dxs[-2] / 2))
        else:
            a_w = a_ws_upwind(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2, dxs[-2]),
                              convection_coefficient(ux_w, dys[-1] + dys[-2] / 2))
            a_s = a_ws_upwind(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2, dys[-2]),
                              convection_coefficient(uy_s, dxs[-1] + dxs[-2] / 2))
        if boundary_right_type == 'D':
            if boundary_top_type == 'D':
                if discretization_type == 'CD':
                    a_b_top = a_en_central_difference(diffusion_coefficient(gamma,
                                                                            dxs[-1] + dxs[-2] / 2, dys[-1]),
                                                      convection_coefficient(uy_b_top, dxs[-1] + dxs[-2] / 2))
                    a_b_right = a_en_central_difference(diffusion_coefficient(gamma,
                                                                              dys[-1] + dys[-2] / 2, dxs[-1]),
                                                        convection_coefficient(ux_b_right, dys[-1] + dys[-2]
                                                                               / 2))
                else:
                    a_b_top = a_en_upwind(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2, dys[-1]),
                                          convection_coefficient(uy_b_top, dxs[-1] + dxs[-2] / 2))
                    a_b_right = a_en_upwind(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2, dxs[-1]),
                                            convection_coefficient(ux_b_right, dys[-1] + dys[-2] / 2))
                aps[i] = ap(a_b_right, a_w, a_b_top, a_s, convection_coefficient(ux_b_right, dys[-1] + dys[-2] / 2),
                            convection_coefficient(ux_w, dys[-1] + dys[-2] / 2),
                            convection_coefficient(uy_b_top, dxs[-1] + dxs[-2] / 2),
                            convection_coefficient(uy_s, dxs[-1] + dxs[-2] / 2))
                b_matrix[i] = dirichlet_dirichlet_boundary_source(a_b_top, a_b_right, boundary_top_value,
                                                                  boundary_right_value)
    elif i < grid_points_y:  # Left boundary
        if velocity_field:
            uy_s = vel_field_y(radius_coordinate(length_x, length_y, dxs[0], np.sum(dys[:i]) - dys[i - 1] / 2),
                               theta_coordinate(length_x, length_y, dxs[0], np.sum(dys[:i]) - dys[i - 1] / 2))
            ux_e = vel_field_x(radius_coordinate(length_x, length_y, dxs[0] + dxs[1] / 2, np.sum(dys[:i])),
                               theta_coordinate(length_x, length_y, dxs[0] + dxs[1] / 2, np.sum(dys[:i])))
            uy_n = vel_field_y(radius_coordinate(length_x, length_y, dxs[0], np.sum(dys[:i]) + dys[i + 1] / 2),
                               theta_coordinate(length_x, length_y, dxs[0], np.sum(dys[:i]) + dys[i + 1] / 2))
            ux_b_left = vel_field_x(radius_coordinate(length_x, length_y, 0, np.sum(dys[:i])),
                                    theta_coordinate(length_x, length_y, 0, np.sum(dys[:i])))
        else:
            uy_s = u_y
            ux_e = u_x
            uy_n = u_y
            ux_b_left = u_x
        if discretization_type == 'CD':
            a_n = a_en_central_difference(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[i + 1]),
                                          convection_coefficient(uy_n, dxs[0] + dxs[1] / 2))
            a_e = a_en_central_difference(diffusion_coefficient(gamma, dys[i + 1] / 2 + dys[i] / 2, dxs[1]),
                                          convection_coefficient(ux_e, dys[i + 1] / 2 + dys[i] / 2))
            a_s = a_ws_central_difference(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[i]),
                                          convection_coefficient(uy_s, dxs[0] + dxs[1] / 2))
        else:
            a_n = a_en_upwind(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[i + 1]),
                              convection_coefficient(uy_n, dxs[0] + dxs[1] / 2))
            a_e = a_en_upwind(diffusion_coefficient(gamma, dys[i + 1] / 2 + dys[i] / 2, dxs[1]),
                              convection_coefficient(ux_e, dys[i + 1] / 2 + dys[i] / 2))
            a_s = a_ws_upwind(diffusion_coefficient(gamma, dxs[0] + dxs[1] / 2, dys[i]),
                              convection_coefficient(uy_s, dxs[0] + dxs[1] / 2))
        if boundary_left_type == 'D':
            if discretization_type == 'CD':
                a_b = a_ws_central_difference(diffusion_coefficient(gamma, dys[i + 1] / 2 + dys[i] / 2,
                                                                    dxs[0]),
                                              convection_coefficient(ux_b_left, dys[i + 1] / 2 + dys[i] / 2))
            else:
                a_b = a_ws_upwind(diffusion_coefficient(gamma, dys[i + 1] / 2 + dys[i] / 2, dxs[0]),
                                  convection_coefficient(ux_b_left, dys[i + 1] / 2 + dys[i] / 2))
            aps[i] = ap(a_e, a_b, a_n, a_s, convection_coefficient(ux_e, dxs[0] + dxs[1] / 2),
                        convection_coefficient(ux_b_left, dys[i + 1] / 2 + dys[i] / 2),
                        convection_coefficient(uy_n, dxs[0] + dxs[1] / 2),
                        convection_coefficient(uy_s, dxs[0] + dxs[1] / 2))
            b_matrix[i] = dirichlet_boundary_source(a_b, boundary_left_value)

    elif i >= grid_points_y * grid_points_x - grid_points_y:  # Right boundary
        if velocity_field:
            uy_s = vel_field_y(radius_coordinate(length_x, length_y, length_x - dxs[-1],
                                                 np.sum(dys[:i - column_counter * grid_points_y])
                                                 - dys[i - column_counter * grid_points_y - 1] / 2),
                               theta_coordinate(length_x, length_y, length_x - dxs[-1],
                                                np.sum(dys[:i - column_counter * grid_points_y])
                                                - dys[i - column_counter * grid_points_y - 1] / 2))
            ux_w = vel_field_x(
                radius_coordinate(length_x, length_y, length_x - dxs[-1] - dxs[-2] / 2,
                                  np.sum(dys[:i - column_counter * grid_points_y])),
                theta_coordinate(length_x, length_y, length_x - dxs[-1] - dxs[-2] / 2,
                                 np.sum(dys[:i - column_counter * grid_points_y])))
            uy_n = vel_field_y(radius_coordinate(length_x, length_y, length_x - dxs[-1],
                                                 np.sum(dys[:i - column_counter * grid_points_y])
                                                 + dys[i - column_counter * grid_points_y + 1] / 2),
                               theta_coordinate(length_x, length_y, length_x - dxs[-1],
                                                np.sum(dys[:i - column_counter * grid_points_y])
                                                + dys[i - column_counter * grid_points_y + 1] / 2))
            ux_b_right = vel_field_x(radius_coordinate(length_x, length_y, length_x,
                                                       np.sum(dys[:i - column_counter * grid_points_y])),
                                     theta_coordinate(length_x, length_y, length_x,
                                                      np.sum(dys[:i - column_counter * grid_points_y])))
        else:
            uy_s = u_y
            ux_w = u_x
            uy_n = u_y
            ux_b_right = u_x
        if discretization_type == 'CD':
            a_n = a_en_central_difference(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2,
                                                                dys[(i - column_counter * grid_points_y) + 1]),
                                          convection_coefficient(uy_n, dxs[-1] + dxs[-2] / 2))
            a_w = a_ws_central_difference(diffusion_coefficient(gamma,
                                                                dys[(i - column_counter * grid_points_y) + 1]
                                                                / 2 + dys[i - column_counter * grid_points_y]
                                                                / 2, dxs[-2]), convection_coefficient(ux_w,
                                                                                                      dys[(
                                                                                                                  i - column_counter * grid_points_y) + 1]
                                                                                                      / 2 + dys[
                                                                                                          i - column_counter * grid_points_y]
                                                                                                      / 2))
            a_s = a_ws_central_difference(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2,
                                                                dys[i - column_counter * grid_points_y]),
                                          convection_coefficient(uy_s, dxs[-1] + dxs[-2] / 2))
        else:
            a_n = a_en_upwind(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2,
                                                    dys[(i - column_counter * grid_points_y) + 1]),
                              convection_coefficient(uy_n, dxs[-1] + dxs[-2] / 2))
            a_w = a_ws_upwind(diffusion_coefficient(gamma, dys[(i - column_counter * grid_points_y) + 1]
                                                    / 2 + dys[i - column_counter * grid_points_y]
                                                    / 2, dxs[-2]), convection_coefficient(ux_w,
                                                                                          dys[(
                                                                                                      i - column_counter * grid_points_y) + 1]
                                                                                          / 2 + dys[
                                                                                              i - column_counter * grid_points_y] / 2))
            a_s = a_ws_upwind(diffusion_coefficient(gamma, dxs[-1] + dxs[-2] / 2,
                                                    dys[i - column_counter * grid_points_y]),
                              convection_coefficient(uy_s, dxs[-1] + dxs[-2] / 2))
        if boundary_right_type == 'D':
            if discretization_type == 'CD':
                a_b = a_en_central_difference(diffusion_coefficient(gamma,
                                                                    dys[(i - column_counter * grid_points_y)
                                                                        + 1] / 2 + dys[i - column_counter *
                                                                                       grid_points_y] / 2,
                                                                    dxs[-1]), convection_coefficient(ux_b_right,
                                                                                                     dys[(
                                                                                                                 i - column_counter * grid_points_y)
                                                                                                         + 1] / 2 + dys[
                                                                                                         i - column_counter *
                                                                                                         grid_points_y] / 2))
            else:
                a_b = a_en_upwind(diffusion_coefficient(gamma,
                                                        dys[(i - column_counter * grid_points_y) + 1]
                                                        / 2 + dys[i - column_counter * grid_points_y] / 2,
                                                        dxs[-1]),
                                  convection_coefficient(ux_b_right, dys[(i - column_counter * grid_points_y)
                                                                         + 1] / 2 + dys[i - column_counter *
                                                                                        grid_points_y] / 2))
            aps[i] = ap(a_b, a_w, a_n, a_s, convection_coefficient(ux_b_right, dys[(i - column_counter * grid_points_y)
                                                                                   + 1] / 2 + dys[i - column_counter *
                                                                                                  grid_points_y] / 2),
                        convection_coefficient(ux_w, dys[(i - column_counter * grid_points_y) + 1]
                                               / 2 + dys[i - column_counter * grid_points_y] / 2),
                        convection_coefficient(uy_n, dxs[-1] + dxs[-2] / 2),
                        convection_coefficient(uy_s, dxs[-1] + dxs[-2] / 2))
            b_matrix[i] = dirichlet_boundary_source(a_b, boundary_right_value)

    elif i % grid_points_y == (grid_points_y - 1):  # Top boundary
        if velocity_field:
            uy_s = vel_field_y(radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
                                                 length_y - dys[-1] - dys[-2] / 2),
                               theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
                                                length_y - dys[-1] - dys[-2] / 2))
            ux_w = vel_field_x(
                radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]) - dxs[column_counter - 1] / 2,
                                  length_y - dys[-1]),
                theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]) - dxs[column_counter - 1] / 2,
                                 length_y - dys[-1]))
            ux_e = vel_field_x(radius_coordinate(length_x, length_y,
                                                 np.sum(dxs[:column_counter]) + dxs[column_counter + 1] / 2,
                                                 length_y - dys[-1]),
                               theta_coordinate(length_x, length_y,
                                                np.sum(dxs[:column_counter]) + dxs[column_counter + 1] / 2,
                                                length_y - dys[-1]))
            uy_b_top = vel_field_y(radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]), length_y),
                                   theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]), length_y))
        else:
            uy_s = u_y
            ux_w = u_x
            ux_e = u_x
            uy_b_top = u_y
        if discretization_type == 'CD':
            a_w = a_ws_central_difference(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2,
                                                                dxs[column_counter]),
                                          convection_coefficient(ux_w, dys[-1] + dys[-2] / 2))
            a_e = a_en_central_difference(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2,
                                                                dxs[column_counter + 1]),
                                          convection_coefficient(ux_e, dys[-1] + dys[-2] / 2))
            a_s = a_ws_central_difference(diffusion_coefficient(gamma, dxs[column_counter] / 2 +
                                                                dxs[column_counter + 1] / 2, dys[-2]),
                                          convection_coefficient(uy_s, dxs[column_counter] / 2 +
                                                                 dxs[column_counter + 1] / 2))
        else:
            a_w = a_ws_upwind(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2, dxs[column_counter]),
                              convection_coefficient(ux_w, dys[-1] + dys[-2] / 2))
            a_e = a_en_upwind(diffusion_coefficient(gamma, dys[-1] + dys[-2] / 2, dxs[column_counter + 1]),
                              convection_coefficient(ux_e, dys[-1] + dys[-2] / 2))
            a_s = a_ws_upwind(diffusion_coefficient(gamma, dxs[column_counter] / 2 +
                                                    dxs[column_counter + 1] / 2, dys[-2]),
                              convection_coefficient(uy_s, dxs[column_counter] / 2 +
                                                     dxs[column_counter + 1] / 2))
        if boundary_top_type == 'D':
            if discretization_type == 'CD':
                a_b = a_en_central_difference(diffusion_coefficient(gamma, dxs[column_counter] / 2 +
                                                                    dxs[column_counter + 1] / 2, dys[-1]),
                                              convection_coefficient(uy_b_top, dxs[column_counter] / 2 +
                                                                     dxs[column_counter + 1] / 2))
            else:
                a_b = a_en_upwind(diffusion_coefficient(gamma, dxs[column_counter] / 2 +
                                                        dxs[column_counter + 1] / 2, dys[-1]),
                                  convection_coefficient(uy_b_top, dxs[column_counter] / 2 +
                                                         dxs[column_counter + 1] / 2))
            aps[i] = ap(a_e, a_w, a_b, a_s, convection_coefficient(ux_e, dys[-1] + dys[-2] / 2),
                        convection_coefficient(ux_w, dys[-1] + dys[-2] / 2),
                        convection_coefficient(uy_b_top, dxs[column_counter] / 2 + dxs[column_counter + 1] / 2),
                        convection_coefficient(uy_s, dxs[column_counter] / 2 + dxs[column_counter + 1] / 2))
            b_matrix[i] = dirichlet_boundary_source(a_b, boundary_top_value)

    elif i % grid_points_y == 0 and i != 0:  # Bottom boundary
        if velocity_field:
            uy_n = vel_field_y(radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
                                                 dys[0] + dys[1] / 2),
                               theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
                                                dys[0] + dys[1] / 2))
            ux_w = vel_field_x(
                radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]) - dxs[column_counter - 1] / 2,
                                  dys[0]),
                theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]) - dxs[column_counter - 1] / 2,
                                 dys[0]))
            ux_e = vel_field_x(radius_coordinate(length_x, length_y,
                                                 np.sum(dxs[:column_counter]) + dxs[column_counter + 1] / 2,
                                                 dys[0]),
                               theta_coordinate(length_x, length_y,
                                                np.sum(dxs[:column_counter]) + dxs[column_counter + 1] / 2,
                                                dys[0]))
            uy_b_bottom = vel_field_y(radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]), 0),
                                      theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]), 0))
        else:
            uy_n = u_y
            ux_w = u_x
            ux_e = u_x
            uy_b_bottom = u_y
        if discretization_type == 'CD':
            a_w = a_ws_central_difference(diffusion_coefficient(gamma, dys[0] + dys[1] / 2,
                                                                dxs[column_counter]),
                                          convection_coefficient(ux_w, dys[0] + dys[1] / 2))
            a_e = a_en_central_difference(diffusion_coefficient(gamma, dys[0] + dys[1] / 2,
                                                                dxs[column_counter + 1]),
                                          convection_coefficient(ux_e, dys[0] + dys[1] / 2))
            a_n = a_en_central_difference(diffusion_coefficient(gamma, dxs[column_counter] / 2 +
                                                                dxs[column_counter + 1] / 2, dys[1]),
                                          convection_coefficient(uy_n, dxs[column_counter] / 2 +
                                                                 dxs[column_counter + 1] / 2))
        else:
            a_w = a_ws_upwind(diffusion_coefficient(gamma, dys[0] + dys[1] / 2, dxs[column_counter]),
                              convection_coefficient(ux_w, dys[0] + dys[1] / 2))
            a_e = a_en_upwind(diffusion_coefficient(gamma, dys[0] + dys[1] / 2, dxs[column_counter + 1]),
                              convection_coefficient(ux_e, dys[0] + dys[1] / 2))
            a_n = a_en_upwind(diffusion_coefficient(gamma, dxs[column_counter] / 2 +
                                                    dxs[column_counter + 1] / 2, dys[1]),
                              convection_coefficient(uy_n, dxs[column_counter] / 2 +
                                                     dxs[column_counter + 1] / 2))
        if boundary_bottom_type == 'D':
            if discretization_type == 'CD':
                a_b = a_ws_central_difference(diffusion_coefficient(gamma, dxs[column_counter] / 2
                                                                    + dxs[column_counter + 1] / 2, dys[0]),
                                              convection_coefficient(uy_b_bottom, dxs[column_counter] / 2
                                                                     + dxs[column_counter + 1] / 2))
            else:
                a_b = a_ws_upwind(diffusion_coefficient(gamma, dxs[column_counter] / 2
                                                        + dxs[column_counter + 1] / 2, dys[0]),
                                  convection_coefficient(uy_b_bottom, dxs[column_counter] / 2
                                                         + dxs[column_counter + 1] / 2))
            aps[i] = ap(a_e, a_w, a_n, a_b, convection_coefficient(ux_e, dys[0] + dys[1] / 2),
                        convection_coefficient(ux_w, dys[0] + dys[1] / 2),
                        convection_coefficient(uy_n, dxs[column_counter] / 2 + dxs[column_counter + 1] / 2),
                        convection_coefficient(uy_b_bottom,
                                               dxs[column_counter] / 2 + dxs[column_counter + 1] / 2))
            b_matrix[i] = dirichlet_boundary_source(a_b, boundary_bottom_value)

    else:  # Interior nodes
        if velocity_field:
            uy_n = vel_field_y(radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
                                                 dys[i - column_counter * grid_points_y] +
                                                 dys[i - column_counter * grid_points_y + 1] / 2),
                               theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
                                                dys[i - column_counter * grid_points_y] +
                                                dys[i - column_counter * grid_points_y + 1] / 2))
            ux_w = vel_field_x(
                radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]) - dxs[column_counter - 1] / 2,
                                  np.sum(dys[:i - column_counter * grid_points_y])),
                theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]) - dxs[column_counter - 1] / 2,
                                 np.sum(dys[:i - column_counter * grid_points_y])))
            ux_e = vel_field_x(radius_coordinate(length_x, length_y,
                                                 np.sum(dxs[:column_counter]) + dxs[column_counter + 1] / 2,
                                                 np.sum(dys[:i - column_counter * grid_points_y])),
                               theta_coordinate(length_x, length_y,
                                                np.sum(dxs[:column_counter]) + dxs[column_counter + 1] / 2,
                                                np.sum(dys[:i - column_counter * grid_points_y])))
            uy_s = vel_field_y(radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
                                                 dys[i - column_counter * grid_points_y] -
                                                 dys[i - column_counter * grid_points_y - 1] / 2),
                               theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
                                                dys[i - column_counter * grid_points_y] -
                                                dys[i - column_counter * grid_points_y - 1] / 2))
        else:
            uy_n = u_y
            ux_w = u_x
            ux_e = u_x
            uy_s = u_y
        if discretization_type == 'CD':
            a_w = a_ws_central_difference(diffusion_coefficient(gamma,
                                                                dys[(i - column_counter * grid_points_y) + 1]
                                                                / 2 + dys[i - column_counter * grid_points_y]
                                                                / 2, dxs[column_counter]),
                                          convection_coefficient(ux_w, dys[(i - column_counter * grid_points_y) + 1] / 2
                                                                 + dys[i - column_counter * grid_points_y] / 2))
            a_e = a_en_central_difference(diffusion_coefficient(gamma,
                                                                dys[(i - column_counter * grid_points_y) + 1]
                                                                / 2 + dys[i - column_counter * grid_points_y]
                                                                / 2, dxs[column_counter + 1]),
                                          convection_coefficient(ux_e,
                                                                 dys[(i - column_counter * grid_points_y) + 1]
                                                                 / 2 + dys[i - column_counter * grid_points_y]
                                                                 / 2))
            a_n = a_en_central_difference(diffusion_coefficient(gamma, dxs[column_counter] / 2
                                                                + dxs[column_counter + 1] / 2,
                                                                dys[(i - column_counter * grid_points_y) + 1]),
                                          convection_coefficient(uy_n, dxs[column_counter] / 2
                                                                 + dxs[column_counter + 1] / 2))
            a_s = a_ws_central_difference(diffusion_coefficient(gamma, dxs[column_counter] / 2
                                                                + dxs[column_counter + 1] / 2,
                                                                dys[i - column_counter * grid_points_y]),
                                          convection_coefficient(uy_s, dxs[column_counter] / 2
                                                                 + dxs[column_counter + 1] / 2))
        else:
            a_w = a_ws_upwind(diffusion_coefficient(gamma,
                                                    dys[(i - column_counter * grid_points_y) + 1]
                                                    / 2 + dys[i - column_counter * grid_points_y]
                                                    / 2, dxs[column_counter]),
                              convection_coefficient(ux_w, dys[(i - column_counter * grid_points_y) + 1] / 2
                                                     + dys[i - column_counter * grid_points_y] / 2))
            a_e = a_en_upwind(diffusion_coefficient(gamma,
                                                    dys[(i - column_counter * grid_points_y) + 1]
                                                    / 2 + dys[i - column_counter * grid_points_y]
                                                    / 2, dxs[column_counter + 1]),
                              convection_coefficient(ux_e, dys[(i - column_counter * grid_points_y) + 1] / 2
                                                     + dys[i - column_counter * grid_points_y] / 2))
            a_n = a_en_upwind(diffusion_coefficient(gamma, dxs[column_counter] / 2
                                                    + dxs[column_counter + 1] / 2,
                                                    dys[(i - column_counter * grid_points_y) + 1]),
                              convection_coefficient(uy_n, dxs[column_counter] / 2
                                                     + dxs[column_counter + 1] / 2))
            a_s = a_ws_upwind(diffusion_coefficient(gamma, dxs[column_counter] / 2
                                                    + dxs[column_counter + 1] / 2,
                                                    dys[i - column_counter * grid_points_y]),
                              convection_coefficient(uy_s, dxs[column_counter] / 2
                                                     + dxs[column_counter + 1] / 2))
        aps[i] = ap(a_e, a_w, a_n, a_s, convection_coefficient(ux_e, dys[(i - column_counter * grid_points_y) + 1]
                                                               / 2 + dys[i - column_counter * grid_points_y] / 2),
                    convection_coefficient(ux_w, dys[(i - column_counter * grid_points_y) + 1]
                                           / 2 + dys[i - column_counter * grid_points_y] / 2),
                    convection_coefficient(uy_n, dxs[column_counter] / 2 + dxs[column_counter + 1] / 2),
                    convection_coefficient(uy_s, dxs[column_counter] / 2 + dxs[column_counter + 1] / 2))
        b_matrix[i] = interior_source()

    if i % grid_points_y == 0:  # New rows
        super_diag[i] = -a_n
        diag[i] = aps[i]
    elif i % grid_points_y == grid_points_y - 1:  # Last entry in a row
        diag[i] = aps[i]
        sub_diag[i] = -a_s
    else:  # All other rows
        super_diag[i] = -a_n
        diag[i] = aps[i]
        sub_diag[i] = -a_s
    if i + grid_points_y < len(aps):
        upper_off_diag[i] = -a_e
    if i - grid_points_y >= 0:
        lower_off_diag[i] = -a_w


# Form A matrix from diagonals:
upper_off_diag = upper_off_diag[0:len(diag) - grid_points_y]
lower_off_diag = lower_off_diag[grid_points_y:]
super_diag = super_diag[0:]
sub_diag = sub_diag[1:]
diagonals = [diag, super_diag, sub_diag, upper_off_diag, lower_off_diag]
a_sparse_matrix = diags(diagonals, [0, 1, -1, grid_points_y, -grid_points_y], format='csc')
t1 = time.time()
print('Time to build matrices: ' + str(t1 - t0))

# Solve Ax = b
temps = spsolve(a_sparse_matrix, b_matrix)
t2 = time.time()
print('Time to solve: ' + str(t2 - t1))
print('The maximum temperature is: ' + str(np.max(temps)))

# Format and save data
nodes, node_x, node_y = two_dim_rectangular_mesh_no_boundary(grid_points_x, grid_points_y, length_x,
                                                             length_x, inflation_factor)
x = node_x
y = node_y
z = np.zeros((grid_points_x, grid_points_y))
# Shape z into correct form for contour plot
for i in range(grid_points_y):
    for j in range(grid_points_x):
        z[j, i] = temps.T[grid_points_y * i + j]
# Write data to file
if save_data and velocity_field:
    np.save('data\\{}x{}grid_{}_phi.npy'.format(grid_points_x, grid_points_y, discretization_type), z)
    np.save('data\\{}x{}grid_{}_x.npy'.format(grid_points_x, grid_points_y, discretization_type), x)
    np.save('data\\{}x{}grid_{}_y.npy'.format(grid_points_x, grid_points_y, discretization_type), y)
if save_data and not velocity_field:
    np.save('data\\falseDiffusion\\{}x{}grid_{}_diag.npy'.format(grid_points_x, grid_points_y, discretization_type), z)

# Phi contour plot
fig2, ax2 = plt.subplots(1, 1)
cp = ax2.contourf(x, y, z)
c_bar = fig2.colorbar(cp)
c_bar.set_label('$\phi$', rotation=180)
ax2.set_xlabel('$x$ [m]')
ax2.set_ylabel('$y$ [m]', rotation=0)
ax2.set_title('Distribution of $\phi$')
plt.tight_layout()
plt.savefig('plots\\{}x{}grid_{}_gamma{}_phi.pdf'.format(grid_points_x, grid_points_y, discretization_type, gamma))
plt.show()

# Phi surface plot
# fig3, ax3 = plt.subplots(1, 1)
# im = ax3.pcolormesh(x, y, z, cmap='jet', shading='auto')
# ax3.set_xlabel('$x$ [m]')
# ax3.set_ylabel('$y$ [m]', rotation=0)
# ax3.set_title('Distribution of $\phi$')
# cbar = fig3.colorbar(im)
# cbar.set_label('$\phi$', rotation=180)
# plt.tight_layout()
# plt.show()

# Node Points Plot
# nodes_bound = two_dim_rectangular_mesh_delta(grid_points_x, grid_points_y, length_x, length_x, inflation_factor)
# fig1 = plt.figure()
# plt.scatter(nodes_bound[:, 0], nodes_bound[:, 1])
# # plt.yscale("log")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Discretized rectangular domain')
# plt.tight_layout()
# plt.savefig('nodes_{}.pdf'.format(inflation_factor))
# plt.show()

# Velocity Field Plot
# vel_field = np.zeros((grid_points_y * grid_points_x, 2))
# column_counter = 0
# for i in range(len(vel_field[:, 0])):
#     if i % grid_points_y == 0:
#         column_counter += 1
#     rad = radius_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
#                             np.sum(dys[:i - grid_points_y * column_counter]))
#     th = theta_coordinate(length_x, length_y, np.sum(dxs[:column_counter]),
#                           np.sum(dys[:i - grid_points_y * column_counter]))
#     vel_field[i, 0] = vel_field_x(rad, th)
#     vel_field[i, 1] = vel_field_y(rad, th)
# print(np.max(vel_field))
# print(np.min(vel_field))
# vField_x = np.zeros((grid_points_x, grid_points_y))
# vField_y = np.zeros((grid_points_x, grid_points_y))
# for i in range(grid_points_y):
#     for j in range(grid_points_x):
#         vField_x[j, i] = vel_field[grid_points_y * i + j, 0]
#         vField_y[j, i] = vel_field[grid_points_y * i + j, 1]
# fig = plt.figure()
# plt.streamplot(x, y, vField_x, vField_y, density=0.5)
# plt.show()