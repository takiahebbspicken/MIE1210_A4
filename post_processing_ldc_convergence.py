import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# folder
t1 = 1
t2 = 2
t3 = 4
t4 = 5
t5 = 5
save_on = False

# Load data
xy_grid1 = np.load('complete_data\\data_{}\\xy_grid.npy'.format(t1))
vel1 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t1))
pressure1 = np.load('complete_data\\data_{}\\vel_final.npy'.format(t1))

xy_grid2 = np.load('complete_data\\data_{}\\xy_grid.npy'.format(t2))
vel2 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t2))
pressure2 = np.load('complete_data\\data_{}\\vel_final.npy'.format(t2))

xy_grid3 = np.load('complete_data\\data_{}\\xy_grid.npy'.format(t3))
vel3 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t3))
pressure3 = np.load('complete_data\\data_{}\\vel_final.npy'.format(t3))

xy_grid4 = np.load('complete_data\\data_{}\\xy_grid.npy'.format(t4))
vel4 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t4))
pressure4 = np.load('complete_data\\data_{}\\vel_final.npy'.format(t4))

xy_grid5 = np.load('complete_data\\data_{}\\xy_grid.npy'.format(t5))
vel5 = np.load('complete_data\\data_{}\\pressure_final.npy'.format(t5))
pressure5 = np.load('complete_data\\data_{}\\vel_final.npy'.format(t5))

max_pressures = [np.max(pressure1), np.max(pressure2), np.max(pressure3), np.max(pressure4), np.max(pressure5)]
max_velocities_x = [np.max(vel1[:, 0]), np.max(vel2[:, 0]), np.max(vel3[:, 0]), np.max(vel4[:, 0]), np.max(vel5[:, 0])]
max_velocities_y = [np.max(vel1[:, 1]), np.max(vel2[:, 1]), np.max(vel3[:, 1]), np.max(vel4[:, 1]), np.max(vel5[:, 1])]
grid_sizes = [xy_grid1[0]**2, xy_grid2[0]**2, xy_grid3[0]**2, xy_grid4[0]**2, xy_grid5[0]**2]

### CONVERGENCE STUDY ###
# Velocities
fig1, ax1 = plt.subplots(1, 1)
x = grid_sizes
ln1, = ax1.plot(x, max_velocities_x, label='Maximum $u$-velocity')
ln2, = ax1.plot(x, max_velocities_y, label='Maximum $v$-velocity')
ax1.set_xlabel('Number of nodes')
ax1.set_ylabel('Maximum velocities', rotation=0)
ax1.set_title('Grid convergence study for maximum velocities')
ax1.legend(handles=[ln1, ln2])
plt.tight_layout()
plt.show()
if save_on:
    plt.savefig('final_plots\\grid_convergence_velocities.pdf')

# Pressure
fig2, ax2 = plt.subplots(1, 1)
ln3, = ax2.plot(x, max_pressures, label='Maximum pressure')
ax2.set_xlabel('Number of nodes')
ax2.set_ylabel('Maximum pressure', rotation=0)
ax2.set_title('Grid convergence study for maximum pressure')
ax2.legend(handles=[ln3])
plt.tight_layout()
plt.show()
if save_on:
    plt.savefig('final_plots\\grid_convergence_pressure.pdf')
