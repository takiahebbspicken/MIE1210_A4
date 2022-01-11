import os
import numpy as np
import matplotlib.pyplot as plt

# folder
# t = '1640363502.5172205'  # 50 x 50 re = 100
# t = '1640375861.3992555'  # 129 x 129 re = 100
t = '1640424075.3634136'  # 320 x 320 re = 200
save_on = True

# Load data
x_coordinates = np.load('complete_data\\data_{}\\x_coordinates.npy'.format(t))
y_coordinates = np.load('complete_data\\data_{}\\y_coordinates.npy'.format(t))
x_grid = np.load('complete_data\\data_{}\\x_grid.npy'.format(t))
y_grid = np.load('complete_data\\data_{}\\y_grid.npy'.format(t))
x_length = np.load('complete_data\\data_{}\\domain_x.npy'.format(t))
y_length = np.load('complete_data\\data_{}\\domain_y.npy'.format(t))
object_position = np.load('complete_data\\data_{}\\obj_position.npy'.format(t))
object_size = np.load('complete_data\\data_{}\\obj_size.npy'.format(t))
residual_u = np.load('complete_data\\data_{}\\residual_u.npy'.format(t))
residual_v = np.load('complete_data\\data_{}\\residual_v.npy'.format(t))
residual_mi = np.load('complete_data\\data_{}\\residual_mi.npy'.format(t))
pressure = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t))
u_vel = np.load('complete_data\\data_{}\\u_vel_final.npy'.format(t))
v_vel = np.load('complete_data\\data_{}\\v_vel_final.npy'.format(t))

object_corners = [(object_position[0], object_position[1]), (object_position[0] + object_size[0]*x_length,
                                                             object_position[1]),
                  (object_position[0] + object_size[0]*x_length, object_position[1] + object_size[1]*y_length),
                  (object_position[0], object_position[1] + object_size[1]*y_length)]
object_x = [object_position[0], object_position[0] + object_size[0]*x_length,
            object_position[0] + object_size[0]*x_length, object_position[0]]
object_y = [object_position[1], object_position[1], object_position[1] + object_size[1]*y_length,
           object_position[1] + object_size[1]*y_length]

line = plt.Polygon(object_corners, closed=None, fill=True, edgecolor='k')

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
ax1.fill(object_x, object_y, facecolor='dimgrey')
ax1.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
ax1.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldcS\\pressure_contour_ldcS.pdf')
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
ax2.fill(object_x, object_y, facecolor='dimgrey')
ax2.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
ax2.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldcS\\uvelocity_contour_ldcS.pdf')
plt.show()


#V velocity contour:
fig3, ax3 = plt.subplots(1, 1)
v_vel = v_vel.reshape((ygrid, xgrid), order='F')
cp3 = plt.contourf(x, y, v_vel)
c_bar3 = fig3.colorbar(cp3)
c_bar3.set_label('$v$-velocity')
ax3.set_xlabel('$x$ [m]')
ax3.set_ylabel('$y$ [m]', rotation=0)
ax3.set_title('$v$-velocity Field')
ax3.fill(object_x, object_y, facecolor='dimgrey')
ax3.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
ax3.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldcS\\vvelocity_contour_ldcS.pdf')
plt.show()


# Combined streamlines and pressure field
fig4, ax4 = plt.subplots(1, 1)
fig4.set_size_inches(8, 6)
cp = plt.contourf(x, y, pressure)
line = plt.Polygon(object_corners, closed=None, fill=True, edgecolor='k')
c_bar = fig4.colorbar(cp)
c_bar.set_label('Pressure')
ax4.set_xlabel('$x$ [m]')
ax4.set_ylabel('$y$ [m]', rotation=0)
ax4.set_title('Pressure and Velocity Fields')
ax4.set_xlim(np.min(x_coordinates), np.max(x_coordinates))
ax4.set_ylim(np.min(y_coordinates), np.max(y_coordinates))
plt.tight_layout()
# Streamline plot
# vel_mag = np.sqrt(u_vel ** 2 + v_vel ** 2)
plt.streamplot(x, y, u_vel, v_vel, color='k', density=3)  # linewidth=vel_mag / vel_mag.max()
ax4.fill(object_x, object_y, facecolor='dimgrey', zorder=5)
if save_on:
    plt.savefig('final_plots\\ldcS\\pressure_velocity_combined_ldcS.pdf')
plt.show()


# Left eddy zoom
fig4_z1, ax4_z1 = plt.subplots(1, 1)
fig4_z1.set_size_inches(8, 6)
cp_z1 = plt.contourf(x[0:int(len(x)/5)], y[int(len(y)/4):int(len(y)/2)], pressure[int(len(y)/4):int(len(y)/2), 0:int(len(x)/5)])
c_bar_z1 = fig4_z1.colorbar(cp_z1)
c_bar_z1.set_label('Pressure')
ax4_z1.set_xlabel('$x$ [m]')
ax4_z1.set_ylabel('$y$ [m]', rotation=0)
ax4_z1.set_title('Pressure and Velocity Fields')
ax4_z1.set_xlim(x[0], x[int(len(x)/5)])
ax4_z1.set_ylim(y[int(len(y)/4)], y[int(len(y)/2)])
plt.tight_layout()
# Streamline plot
plt.streamplot(x[0:int(len(x)/5)], y[int(len(y)/4):int(len(y)/2)], u_vel[int(len(y)/4):int(len(y)/2), 0:int(len(x)/5)],
               v_vel[int(len(y)/4):int(len(y)/2), 0:int(len(x)/5)], color='k', density=3)
ax4_z1.fill(object_x, object_y, facecolor='dimgrey', zorder=5)
if save_on:
    plt.savefig('final_plots\\ldcS\\pressure_velocity_combined_LE_ldcS.pdf')
plt.show()


# Right eddy
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
ax4_z2.fill(object_x, object_y, facecolor='dimgrey', zorder=5)
if save_on:
    plt.savefig('final_plots\\ldcS\\pressure_velocity_combined_RE_ldcS.pdf')
plt.show()


# Middle eddy
fig4_z3, ax4_z3 = plt.subplots(1, 1)
fig4_z3.set_size_inches(8, 6)
cp_z3 = plt.contourf(x[int(len(x)/4):int(len(x)/2)], y[0:int(len(y)/4)], pressure[0:int(len(y)/4), int(len(x)/4):int(len(x)/2)])
c_bar_z3 = fig4_z3.colorbar(cp_z3)
c_bar_z3.set_label('Pressure')
ax4_z3.set_xlabel('$x$ [m]')
ax4_z3.set_ylabel('$y$ [m]', rotation=0)
ax4_z3.set_title('Pressure and Velocity Fields')
ax4_z3.set_xlim(x[int(len(x)/4)], x[int(len(x)/2)])
ax4_z3.set_ylim(y[0], y[int(len(y)/4)])
plt.tight_layout()
# Streamline plot
plt.streamplot(x[int(len(x)/4):int(len(x)/2)], y[0:int(len(y)/4)], u_vel[0:int(len(y)/4), int(len(x)/4):int(len(x)/2)],
               v_vel[0:int(len(y)/4), int(len(x)/4):int(len(x)/2)], color='k', density=3)
ax4_z3.fill(object_x, object_y, facecolor='dimgrey', zorder=5)
if save_on:
    plt.savefig('final_plots\\ldcS\\pressure_velocity_combined_ME_ldcS.pdf')
plt.show()



### CONVERGENCE PLOTS ###
fig5, ax5 = plt.subplots(1, 1)
x_conv = np.arange(len(residual_u))
residuals_u = residual_u
residuals_v = residual_v
mass_imbalance = residual_mi
ln1, = ax5.plot(x_conv, residuals_u, label='$u$-velocity residuals')
ln2, = ax5.plot(x_conv, residuals_v, label='$v$-velocity residuals')
ln3, = ax5.plot(x_conv, mass_imbalance, label='Mass imbalance')
ax5.set_xlabel('Iteration')
ax5.set_ylabel('Residuals', rotation=0)
ax5.set_title('Residuals over code convergence')
ax5.legend(handles=[ln1, ln2, ln3])
ax5.set_yscale('log')
plt.tight_layout()
if save_on:
    plt.savefig('final_plots\\ldcS\\convergence_ldcS.pdf')
plt.show()

