import numpy as np
import matplotlib.pyplot as plt

t = '1641937571.7534337'  # 80 x 80
save_on = False

# Load data
residual_u = np.load('complete_data\\data_{}\\residual_u.npy'.format(t))
residual_v = np.load('complete_data\\data_{}\\residual_v.npy'.format(t))
residual_mi = np.load('complete_data\\data_{}\\residual_mi.npy'.format(t))

x_coordinates = np.load('complete_data\\data_{}\\x_coordinates.npy'.format(t))
y_coordinates = np.load('complete_data\\data_{}\\y_coordinates.npy'.format(t))
x_grid = np.load('complete_data\\data_{}\\x_grid.npy'.format(t))
y_grid = np.load('complete_data\\data_{}\\y_grid.npy'.format(t))

pressure = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t))
u_vel = np.load('complete_data\\data_{}\\u_vel_final.npy'.format(t))
v_vel = np.load('complete_data\\data_{}\\v_vel_final.npy'.format(t))

# pressure = np.load('complete_data\\data_{}\\pressure_iteration_0.npy'.format(t))
# u_vel = np.load('complete_data\\data_{}\\u_vel_iteration_0.npy'.format(t))
# v_vel = np.load('complete_data\\data_{}\\v_vel_iteration_0.npy'.format(t))

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
x = x_coordinates
y = y_coordinates
xgrid = x_grid
ygrid = y_grid
pressure = pressure.reshape((ygrid, xgrid), order='F')
cp = plt.contourf(x, y, pressure)
c_bar = fig1.colorbar(cp)
c_bar.set_label('Pressure')
ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$y$ [m]', rotation=0)
ax1.set_title('Pressure Field')
ax1.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
ax1.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldc\\pressure_contour_ldc.pdf')
plt.show()


# U velocity contour:
fig2, ax2 = plt.subplots(1, 1)
u_vel = u_vel.reshape((ygrid, xgrid), order='F')
cp2 = plt.contourf(x, y, u_vel)
c_bar2 = fig2.colorbar(cp2)
c_bar2.set_label('$u$-velocity')
ax2.set_xlabel('$x$ [m]')
ax2.set_ylabel('$y$ [m]', rotation=0)
ax2.set_title('$u$-velocity Field')
ax2.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
ax2.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldc\\uvelocity_contour_ldc.pdf')
plt.show()
print('Max u {}'.format(np.max(u_vel)))
print(np.allclose(u_vel, -u_vel.T, rtol=1e-5, atol=1e-8))

#V velocity contour:
fig3, ax3 = plt.subplots(1, 1)
v_vel = v_vel.reshape((ygrid, xgrid), order='F')
cp3 = plt.contourf(x, y, v_vel)
c_bar3 = fig3.colorbar(cp3)
c_bar3.set_label('$v$-velocity')
ax3.set_xlabel('$x$ [m]')
ax3.set_ylabel('$y$ [m]', rotation=0)
ax3.set_title('$v$-velocity Field')
ax3.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
ax3.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldc\\vvelocity_contour_ldc.pdf')
plt.show()

print(np.allclose(v_vel, -v_vel.T, rtol=1e-5, atol=1e-8))

# Combined streamlines and pressure field
fig4, ax4 = plt.subplots(1, 1)
fig4.set_size_inches(8, 6)
cp = plt.contourf(x, y, pressure)
c_bar = fig4.colorbar(cp)
c_bar.set_label('Pressure')
ax4.set_xlabel('$x$ [m]')
ax4.set_ylabel('$y$ [m]', rotation=0)
ax4.set_title('Pressure and Velocity Fields')
ax4.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
ax4.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
plt.tight_layout()
# Streamline plot
vel_mag = np.sqrt(u_vel ** 2 + v_vel ** 2).reshape((ygrid, xgrid))
plt.streamplot(x, y, u_vel, v_vel, color='k', density=5)
if save_on:
    plt.savefig('final_plots\\ldc\\pressure_velocity_combined_ldc.pdf')
plt.show()


fig4_z1, ax4_z1 = plt.subplots(1, 1)
fig4_z1.set_size_inches(8, 6)
cp_z1 = plt.contourf(x[0:int(len(x)/4)], y[0:int(len(y)/4)], pressure[0:int(len(x)/4), 0:int(len(y)/4)])
c_bar_z1 = fig4_z1.colorbar(cp_z1)
c_bar_z1.set_label('Pressure')
ax4_z1.set_xlabel('$x$ [m]')
ax4_z1.set_ylabel('$y$ [m]', rotation=0)
ax4_z1.set_title('Pressure and Velocity Fields')
ax4_z1.set_xlim(x[0], x[int(len(x)/4)])
ax4_z1.set_ylim(y[0], y[int(len(y)/4)])
plt.tight_layout()
# Streamline plot
plt.streamplot(x[0:int(len(x)/4)], y[0:int(len(y)/4)], u_vel[0:int(len(x)/4), 0:int(len(y)/4)],
               v_vel[0:int(len(x)/4), 0:int(len(y)/4)], color='k', density=3)
if save_on:
    plt.savefig('final_plots\\ldc\\pressure_velocity_combined_LE_ldc.pdf')
plt.show()

fig4_z2, ax4_z2 = plt.subplots(1, 1)
fig4_z2.set_size_inches(8, 6)
cp_z2 = plt.contourf(x[int(np.ceil(len(x) - len(x)/4)):], y[0:int(len(y)/4)], pressure[0:int(len(y)/4),
                                                                     int(np.ceil(len(x) - len(x)/4)):])
c_bar_z2 = fig4_z2.colorbar(cp_z2)
c_bar_z2.set_label('Pressure')
ax4_z2.set_xlabel('$x$ [m]')
ax4_z2.set_ylabel('$y$ [m]', rotation=0)
ax4_z2.set_title('Pressure and Velocity Fields')
ax4_z2.set_xlim(x[int(np.ceil(len(x) - len(x)/4))], x[-1])
ax4_z2.set_ylim(y[0], y[int(len(y)/4)])
plt.tight_layout()
# Streamline plot
plt.streamplot(x[int(np.ceil(len(x) - len(x)/4)):], y[0:int(len(y)/4)], u_vel[0:int(len(y)/4), int(np.ceil(len(x) - len(x)/4)):],
               v_vel[0:int(len(y)/4), int(np.ceil(len(x) - len(x)/4)):], color='k', density=3)
if save_on:
    plt.savefig('final_plots\\ldc\\pressure_velocity_combined_RE_ldc.pdf')
plt.show()


### VELOCITY PROFILE PLOTS ###
# u velocity vertical
fig5, ax5 = plt.subplots(1, 1)
u_center_vertical = u_vel[:, int(np.floor(len(u_vel[0, :]) / 2))]
ln1, = ax5.plot(y, u_center_vertical, label='$u$-velocity results')
scatter1 = ax5.scatter(ghia_u_vertical[0, :], ghia_u_vertical[1, :], color='k', label='Ghia et al. results')
ax5.set_xlabel('Vertical distance along center')
ax5.set_ylabel('$u$-velocity', rotation=0)
ax5.yaxis.labelpad = 20
ax5.set_title('$u$-velocity profile vertically along center of cavity')
ax5.legend(handles=[ln1, scatter1])
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldc\\uvelocity_profile_ldc.pdf')
plt.show()


# v velocity horizontal
fig6, ax6 = plt.subplots(1, 1)
v_center_horizontal = v_vel[int(np.floor(len(u_vel[0, :]) / 2)), :]
scatter2 = ax6.scatter(ghia_v_horizontal[0, :], ghia_v_horizontal[1, :], color='k', label='Ghia et al. results')
ln2, = ax6.plot(x, v_center_horizontal, label='$v$-velocity results')
ax6.set_xlabel('Horizontal distance along center')
ax6.set_ylabel('$v$-velocity', rotation=0)
ax6.yaxis.labelpad = 20
ax6.set_title('$v$-velocity profile horizontally along center of cavity')
ax6.legend(handles=[ln2, scatter2])
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldc\\vvelocity_profile_ldc.pdf')
plt.show()


### CONVERGENCE PLOTS ###
fig7, ax7 = plt.subplots(1, 1)
x_conv = np.arange(len(residual_u))
residuals_u = residual_u
residuals_v = residual_v
if len(residual_mi) == 2 * len(residuals_u):
    mass_imbalance = residual_mi[1::2]
else:
    mass_imbalance = residual_mi
ln3, = ax7.plot(x_conv, residuals_u, label='$u$-velocity residuals')
ln4, = ax7.plot(x_conv, residuals_v, label='$v$-velocity residuals')
ln5, = ax7.plot(x_conv, mass_imbalance, label='Mass imbalance')
ax7.set_xlabel('Iteration')
ax7.set_ylabel('Residuals', rotation=0)
ax7.yaxis.labelpad = 20
ax7.set_title('Residuals over code convergence')
ax7.legend(handles=[ln3, ln4, ln5])
ax7.set_yscale('log')
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldc\\convergence_ldc.pdf')
plt.show()

