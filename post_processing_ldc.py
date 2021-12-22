import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# folder
t = 1

# Load data
xy_coordinates = np.load('complete_data\\data_{}\\xy_coordinates.npy'.format(t))
xy_grid = np.load('complete_data\\data_{}\\xy_grid.npy'.format(t))
residuals = np.load('complete_data\\data_{}\\residuals.npy'.format(t))
vel = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t))
pressure = np.load('complete_data\\data_{}\\vel_final.npy'.format(t))

# Results from Ghia et al.
ghia_v_horizontal = np.array([[1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563,
                              0.0938, 0.0781, 0.0703, 0.0625, 0.0],
                             [0.0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454,
                              0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.000]])
ghia_u_vertical = np.array([[1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.500, 0.4531, 0.2831, 0.1719,
                            0.1016, 0.0703, 0.0625, 0.0547, 0.0],
                           [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.0032, -0.13641, -0.20581, -0.21090,
                            -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0]])


### CONTOUR PLOTS ###
# Pressure contour:
fig1, ax1 = plt.subplots(1, 1)
x = xy_coordinates[:, 0]
y = xy_coordinates[:, 1]
xgrid = xy_grid[0]
ygrid = xy_grid[1]
pressure = pressure.reshape((ygrid, xgrid), order='F')
cp = plt.contourf(x, y, pressure)
c_bar = fig1.colorbar(cp)
c_bar.set_label('Pressure')
ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$y$ [m]', rotation=0)
ax1.set_title('Pressure Field')
plt.tight_layout()
plt.show()

# U velocity contour:
fig2, ax2 = plt.subplots(1, 1)
u_vel = vel[:, 0].reshape((ygrid, xgrid), order='F')
cp2 = plt.contourf(x, y, u_vel)
c_bar2 = fig2.colorbar(cp2)
c_bar2.set_label('$u$-velocity')
ax2.set_xlabel('$x$ [m]')
ax2.set_ylabel('$y$ [m]', rotation=0)
ax2.set_title('$u$-velocity Field')
plt.tight_layout()
plt.show()

#V velocity contour:
fig3, ax3 = plt.subplots(1, 1)
v_vel = vel[:, 1].reshape((ygrid, xgrid), order='F')
cp3 = plt.contourf(x, y, v_vel)
c_bar3 = fig2.colorbar(cp2)
c_bar3.set_label('$v$-velocity')
ax3.set_xlabel('$x$ [m]')
ax3.set_ylabel('$y$ [m]', rotation=0)
ax3.set_title('$v$-velocity Field')
plt.tight_layout()
plt.show()

# Combined streamlines and pressure field
fig4, ax4 = plt.subplots(1, 1)
cp = plt.contourf(x, y, pressure)
c_bar = fig4.colorbar(cp)
c_bar.set_label('Pressure')
ax4.set_xlabel('$x$ [m]')
ax4.set_ylabel('$y$ [m]', rotation=0)
ax4.set_title('Pressure and Velocity Fields')
plt.tight_layout()
# Streamline plot
vel_mag = np.sqrt(u_vel[:, 0] ** 2 + v_vel[:, 1] ** 2).reshape((ygrid, xgrid))
plt.streamplot(x, y, u_vel, v_vel, density=0.5)  # linewidth=vel_mag / vel_mag.max()
plt.show()

### VELOCITY PROFILE PLOTS ###
# u velocity vertical
fig5, ax5 = plt.subplots(1, 1)
u_center_vertical = u_vel[:, u_vel.size // 2]
ln1, = ax5.plot(y, u_center_vertical, label='$u$-velocity results')
scatter1 = ax5.scatter(ghia_u_vertical[0, :], ghia_u_vertical[1, :], label='Ghia et al. results')
ax5.set_xlabel('Vertical distance along center')
ax5.set_ylabel('$u$-velocity', rotation=0)
ax5.set_title('$u$-velocity profile vertically along center of cavity')
ax5.legend(handles=[ln1, scatter1])
plt.tight_layout()
plt.show()

# v velocity horizontal
fig6, ax6 = plt.subplots(1, 1)
v_center_horizontal = u_vel[:, u_vel.size // 2]
scatter2 = ax6.scatter(ghia_v_horizontal[0, :], ghia_v_horizontal[1, :], label='Ghia et al. results')
ln2, = ax6.plot(x, v_center_horizontal, label='$v$-velocity results')
ax6.set_xlabel('Horizontal distance along center')
ax6.set_ylabel('$v$-velocity', rotation=0)
ax6.set_title('$v$-velocity profile horizontally along center of cavity')
ax6.legend(handles=[ln2, scatter2])
plt.tight_layout()
plt.show()

### CONVERGENCE PLOTS ###
fig7, ax7 = plt.subplots(1, 1)
x_conv = np.arrange(len(residuals[0, :]))
residuals_u = residuals[:, 0]
residuals_v = residuals[:, 1]
mass_imbalance = residuals[:, 2]
ln3, = ax6.plot(x_conv, residuals_u, label='$u$-velocity residuals')
ln4, = ax6.plot(x_conv, residuals_v, label='$v$-velocity residuals')
ln5, = ax6.plot(x_conv, mass_imbalance, label='Mass imbalance')
ax7.set_xlabel('Iteration')
ax7.set_ylabel('Residuals', rotation=0)
ax7.set_title('Residuals over code convergence')
ax7.legend(handles=[ln3, ln4, ln5])
plt.tight_layout()
plt.show()
