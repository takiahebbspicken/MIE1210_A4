import os
import numpy as np
import matplotlib.pyplot as plt

# folder
t1 = '1640299920.3067224'  # 50 x 50 relaxation everywhere, re = 100, 5000 iterations
t2 = '1640572749.8869405'  # 129 x 129 relax internally, reference pressure, re = 100, 1500 iteration
t3 = '1640500873.9296625'  # 200 x 200
t4 = '1640353884.984672'  # 259 x 259 relaxation everywhere, pressure correction not forced, re = 100, 2000 iterations
t5 = '1640489128.3164742'  # 320 x 320 relaxation everywhere, pressure correction not forced, re = 100, 2000 iterations
save_on = False

# Load data
pressure1 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t1))
u_vel1 = np.load('complete_data\\data_{}\\u_vel_final.npy'.format(t1))
v_vel1 = np.load('complete_data\\data_{}\\v_vel_final.npy'.format(t1))
x_coordinates1 = np.load('complete_data\\data_{}\\x_coordinates.npy'.format(t1))
y_coordinates1 = np.load('complete_data\\data_{}\\y_coordinates.npy'.format(t1))
x_grid1 = np.load('complete_data\\data_{}\\x_grid.npy'.format(t1))
y_grid1 = np.load('complete_data\\data_{}\\y_grid.npy'.format(t1))

pressure2 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t2))
u_vel2 = np.load('complete_data\\data_{}\\u_vel_final.npy'.format(t2))
v_vel2 = np.load('complete_data\\data_{}\\v_vel_final.npy'.format(t2))
x_coordinates2 = np.load('complete_data\\data_{}\\x_coordinates.npy'.format(t2))
y_coordinates2 = np.load('complete_data\\data_{}\\y_coordinates.npy'.format(t2))
x_grid2 = np.load('complete_data\\data_{}\\x_grid.npy'.format(t2))
y_grid2 = np.load('complete_data\\data_{}\\y_grid.npy'.format(t2))

pressure3 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t3))
u_vel3 = np.load('complete_data\\data_{}\\u_vel_final.npy'.format(t3))
v_vel3 = np.load('complete_data\\data_{}\\v_vel_final.npy'.format(t3))
x_coordinates3 = np.load('complete_data\\data_{}\\x_coordinates.npy'.format(t3))
y_coordinates3 = np.load('complete_data\\data_{}\\y_coordinates.npy'.format(t3))
x_grid3 = np.load('complete_data\\data_{}\\x_grid.npy'.format(t3))
y_grid3 = np.load('complete_data\\data_{}\\y_grid.npy'.format(t3))

pressure4 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t4))
u_vel4 = np.load('complete_data\\data_{}\\u_vel_final.npy'.format(t4))
v_vel4 = np.load('complete_data\\data_{}\\v_vel_final.npy'.format(t4))
x_coordinates4 = np.load('complete_data\\data_{}\\x_coordinates.npy'.format(t4))
y_coordinates4 = np.load('complete_data\\data_{}\\y_coordinates.npy'.format(t4))
x_grid4 = np.load('complete_data\\data_{}\\x_grid.npy'.format(t4))
y_grid4 = np.load('complete_data\\data_{}\\y_grid.npy'.format(t4))

pressure5 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t5))
u_vel5 = np.load('complete_data\\data_{}\\u_vel_final.npy'.format(t5))
v_vel5 = np.load('complete_data\\data_{}\\v_vel_final.npy'.format(t5))
x_coordinates5 = np.load('complete_data\\data_{}\\x_coordinates.npy'.format(t5))
y_coordinates5 = np.load('complete_data\\data_{}\\y_coordinates.npy'.format(t5))
x_grid5 = np.load('complete_data\\data_{}\\x_grid.npy'.format(t5))
y_grid5 = np.load('complete_data\\data_{}\\y_grid.npy'.format(t5))

max_pressures = [np.max(pressure1), np.max(pressure2), np.max(pressure3), np.max(pressure4), np.max(pressure5)]
max_velocities_x = [np.max(u_vel1), np.max(u_vel2), np.max(u_vel3), np.max(u_vel4), np.max(u_vel5)]
print(max_velocities_x, (max_velocities_x[-1] - max_velocities_x[-2])/max_velocities_x[-2])
max_velocities_y = [np.max(v_vel1), np.max(v_vel2), np.max(v_vel3), np.max(v_vel4), np.max(v_vel5)]
print(max_velocities_y, (max_velocities_y[-1] - max_velocities_y[-2])/max_velocities_y[-2])
grid_sizes = [x_grid1 * y_grid1, x_grid2 * y_grid2, x_grid3 * y_grid3, x_grid4 * y_grid4, x_grid5 * y_grid5]

u_vel_sq1 = u_vel1.reshape((y_grid1, x_grid1), order='F')
u_vel_sq2 = u_vel2.reshape((y_grid2, x_grid2), order='F')
u_vel_sq3 = u_vel3.reshape((y_grid3, x_grid3), order='F')
u_vel_sq4 = u_vel4.reshape((y_grid4, x_grid4), order='F')
u_vel_sq5 = u_vel5.reshape((y_grid5, x_grid5), order='F')
u_center_vertical1 = u_vel_sq1[:, int(np.floor(len(u_vel_sq1[0, :]) / 2))]
u_center_vertical2 = u_vel_sq2[:, int(np.floor(len(u_vel_sq2[0, :]) / 2))]
u_center_vertical3 = u_vel_sq3[:, int(np.floor(len(u_vel_sq3[0, :]) / 2))]
u_center_vertical4 = u_vel_sq4[:, int(np.floor(len(u_vel_sq4[0, :]) / 2))]
u_center_vertical5 = u_vel_sq5[:, int(np.floor(len(u_vel_sq5[0, :]) / 2))]

v_vel_sq1 = v_vel1.reshape((y_grid1, x_grid1), order='F')
v_vel_sq2 = v_vel2.reshape((y_grid2, x_grid2), order='F')
v_vel_sq3 = v_vel3.reshape((y_grid3, x_grid3), order='F')
v_vel_sq4 = v_vel4.reshape((y_grid4, x_grid4), order='F')
v_vel_sq5 = v_vel5.reshape((y_grid5, x_grid5), order='F')
v_center_horizontal1 = v_vel_sq1[int(np.floor(len(v_vel_sq1[0, :]) / 2)), :]
v_center_horizontal2 = v_vel_sq2[int(np.floor(len(v_vel_sq2[0, :]) / 2)), :]
v_center_horizontal3 = v_vel_sq3[int(np.floor(len(v_vel_sq3[0, :]) / 2)), :]
v_center_horizontal4 = v_vel_sq4[int(np.floor(len(v_vel_sq4[0, :]) / 2)), :]
v_center_horizontal5 = v_vel_sq5[int(np.floor(len(v_vel_sq5[0, :]) / 2)), :]

### CONVERGENCE STUDY ###
# Velocities
fig1, ax1 = plt.subplots(1, 1)
x = grid_sizes
ln1, = ax1.plot(x, max_velocities_x, label='Maximum $u$-velocity')
ln2, = ax1.plot(x, max_velocities_y, label='Maximum $v$-velocity')
ax1.set_xlabel('Number of nodes')
ax1.set_ylabel('Maximum velocities', rotation=0)
ax1.yaxis.labelpad = 50
ax1.set_title('Grid convergence study for maximum velocities')
ax1.legend(handles=[ln1, ln2])
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\grid_convergence_velocities.pdf')
plt.show()


# Pressure
fig2, ax2 = plt.subplots(1, 1)
ln3, = ax2.plot(x, max_pressures, label='Maximum pressure')
ax2.set_xlabel('Number of nodes')
ax2.set_ylabel('Maximum pressure', rotation=0)
ax2.yaxis.labelpad = 40
ax2.set_title('Grid convergence study for maximum pressure')
ax2.legend(handles=[ln3])
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\grid_convergence_pressure.pdf')
plt.show()


# Velocity profiles
ghia_v_horizontal = np.array([[1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563,
                              0.0938, 0.0781, 0.0703, 0.0625, 0.0],
                             [0.0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454,
                              0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.000]])
ghia_u_vertical = np.array([[1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.500, 0.4531, 0.2831, 0.1719,
                            0.1016, 0.0703, 0.0625, 0.0547, 0.0],
                           [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.0032, -0.13641, -0.20581, -0.21090,
                            -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0]])
# u velocity vertical
fig3, ax3 = plt.subplots(1, 1)
ln1, = ax3.plot(y_coordinates1, u_center_vertical1, label='{} nodes'.format(x_grid1 * y_grid1))
ln2, = ax3.plot(y_coordinates2, u_center_vertical2, label='{} nodes'.format(x_grid2 * y_grid2))
ln3, = ax3.plot(y_coordinates3, u_center_vertical3, label='{} nodes'.format(x_grid3 * y_grid3))
ln4, = ax3.plot(y_coordinates4, u_center_vertical4, label='{} nodes'.format(x_grid4 * y_grid4))
ln5, = ax3.plot(y_coordinates5, u_center_vertical5, label='{} nodes'.format(x_grid5 * y_grid5), color='k')
scatter1 = ax3.scatter(ghia_u_vertical[0, :], ghia_u_vertical[1, :], label='Ghia et al. results')
ax3.set_xlabel('Vertical distance along center')
ax3.set_ylabel('$u$-velocity', rotation=0)
ax3.yaxis.labelpad = 20
ax3.set_title('$u$-velocity profile vertically along center of cavity')
ax3.legend(handles=[ln1, ln2, ln3, ln4, ln5, scatter1])
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\uvelocity_profile_convergence.pdf')
plt.show()

# v velocity horizontal
fig4, ax4 = plt.subplots(1, 1)
scatter2 = ax4.scatter(ghia_v_horizontal[0, :], ghia_v_horizontal[1, :], color='k', label='Ghia et al. results')
ln6, = ax4.plot(x_coordinates1, v_center_horizontal1, label='{} nodes'.format(x_grid1 * y_grid1))
ln7, = ax4.plot(x_coordinates2, v_center_horizontal2, label='{} nodes'.format(x_grid2 * y_grid2))
ln8, = ax4.plot(x_coordinates3, v_center_horizontal3, label='{} nodes'.format(x_grid3 * y_grid3))
ln9, = ax4.plot(x_coordinates4, v_center_horizontal4, label='{} nodes'.format(x_grid4 * y_grid4))
ln10, = ax4.plot(x_coordinates5, v_center_horizontal5, label='{} nodes'.format(x_grid5 * y_grid5))
ax4.set_xlabel('Horizontal distance along center')
ax4.set_ylabel('$v$-velocity', rotation=0)
ax4.yaxis.labelpad = 20
ax4.set_title('$v$-velocity profile horizontally along center of cavity')
ax4.legend(handles=[ln6, ln7, ln8, ln9, ln10, scatter2])
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\vvelocity_profile_convergence.pdf')
plt.show()
