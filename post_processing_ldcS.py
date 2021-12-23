import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# folder
t = 1
save_on = False

# Load data
x_coordinates = np.load('complete_data\\data_{}\\x_coordinates.npy'.format(t))
y_coordinates = np.load('complete_data\\data_{}\\y_coordinates.npy'.format(t))
x_grid = np.load('complete_data\\data_{}\\x_grid.npy'.format(t))
y_grid = np.load('complete_data\\data_{}\\y_grid.npy'.format(t))
residual_u = np.load('complete_data\\data_{}\\residual_u.npy'.format(t))
residual_v = np.load('complete_data\\data_{}\\residual_v.npy'.format(t))
residual_mi = np.load('complete_data\\data_{}\\residual_mi.npy'.format(t))
pressure = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t))
u_vel = np.load('complete_data\\data_{}\\u_vel_final.npy'.format(t))
v_vel = np.load('complete_data\\data_{}\\v_vel_final.npy'.format(t))



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
plt.tight_layout()
plt.show()
if save_on:
    plt.savefig('final_plots\\ldcS\\pressure_contour_ldcS.pdf')

# U velocity contour:
fig2, ax2 = plt.subplots(1, 1)
u_vel = u_vel.reshape((ygrid, xgrid), order='F')
cp2 = plt.contourf(x, y, u_vel)
c_bar2 = fig2.colorbar(cp2)
c_bar2.set_label('$u$-velocity')
ax2.set_xlabel('$x$ [m]')
ax2.set_ylabel('$y$ [m]', rotation=0)
ax2.set_title('$u$-velocity Field')
plt.tight_layout()
plt.show()
if save_on:
    plt.savefig('final_plots\\ldcS\\uvelocity_contour_ldcS.pdf')

#V velocity contour:
fig3, ax3 = plt.subplots(1, 1)
v_vel = v_vel.reshape((ygrid, xgrid), order='F')
cp3 = plt.contourf(x, y, v_vel)
c_bar3 = fig2.colorbar(cp2)
c_bar3.set_label('$v$-velocity')
ax3.set_xlabel('$x$ [m]')
ax3.set_ylabel('$y$ [m]', rotation=0)
ax3.set_title('$v$-velocity Field')
plt.tight_layout()
plt.show()
if save_on:
    plt.savefig('final_plots\\ldcS\\vvelocity_contour_ldcS.pdf')

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
if save_on:
    plt.savefig('final_plots\\ldcS\\pressure_velocity_combined_ldcS.pdf')

### CONVERGENCE PLOTS ###
fig5, ax5 = plt.subplots(1, 1)
x_conv = np.arrange(len(residual_u))
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
plt.tight_layout()
plt.show()
if save_on:
    plt.savefig('final_plots\\ldcS\\convergence_ldcS.pdf')
