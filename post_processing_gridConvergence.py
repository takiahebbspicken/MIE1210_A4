import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
# Specify grid sizes for convergence calculation
grid_fine = '320x320'
grid_mid = '160x160'
grid_coarse = '80x80'
# Lengths of the domain
length_x = 1  # Length of x domain in [m]
length_y = 1  # Length of y domain in [m]
# Inflation factor for convergence calculation
inflation_factor = 1.1

def norm_l2(val1, val2):
    # L2 norm equation
    n = len(val1)*len(val1)
    return np.sqrt((1/n)*np.sum((val1-val2)**2))


def order_conv(error_coarse, error_fine, s_coarse, s_fine):
    # Order of convergence calculation
    return np.log((error_coarse/error_fine))/(np.log(s_coarse/s_fine))


# Load data from files
for filename in os.listdir('data\\'):
    # Fine grid CD data
    if grid_fine in filename and '_phi' in filename:
        phi_fine = np.load('data\\{}grid_CD_phi.npy'.format(grid_fine))
    elif grid_fine in filename and '_x' in filename:
        x_fine = np.load('data\\{}grid_CD_x.npy'.format(grid_fine))
    elif grid_fine in filename and '_y' in filename:
        y_fine = np.load('data\\{}grid_CD_y.npy'.format(grid_fine))

    # Mid grid CD data
    if grid_mid in filename and '_phi' in filename:
        phi_mid = np.load('data\\{}grid_CD_phi.npy'.format(grid_mid))
    elif grid_mid in filename and '_x' in filename:
        x_mid = np.load('data\\{}grid_CD_x.npy'.format(grid_mid))
    elif grid_mid in filename and '_y' in filename:
        y_mid = np.load('data\\{}grid_CD_y.npy'.format(grid_mid))

    # Coarse grid CD data
    if grid_coarse in filename and '_phi' in filename:
        phi_coarse = np.load('data\\{}grid_CD_phi.npy'.format(grid_coarse))
    elif grid_coarse in filename and '_x' in filename:
        x_coarse = np.load('data\\{}grid_CD_x.npy'.format(grid_coarse))
    elif grid_coarse in filename and '_y' in filename:
        y_coarse = np.load('data\\{}grid_CD_y.npy'.format(grid_coarse))

    # Fine grid UW data
    if grid_fine in filename and '_phi' in filename and 'UW' in filename:
        phi_fine_inflated = np.load('data\\{}grid_UW_phi.npy'.format(grid_fine, inflation_factor))
    elif grid_fine in filename and '_x' in filename and 'UW' in filename:
        x_fine_inflated = np.load('data\\{}grid_UW_x.npy'.format(grid_fine, inflation_factor))
    elif grid_fine in filename and '_y' in filename and 'UW' in filename:
        y_fine_inflated = np.load('data\\{}grid_UW_y.npy'.format(grid_fine, inflation_factor))

    # Mid grid UW data
    if grid_mid in filename and '_phi' in filename and 'UW' in filename:
        phi_mid_inflated = np.load('data\\{}grid_UW_phi.npy'.format(grid_mid, inflation_factor))
    elif grid_mid in filename and '_x' in filename and 'UW' in filename:
        x_mid_inflated = np.load('data\\{}grid_UW_x.npy'.format(grid_mid, inflation_factor))
    elif grid_mid in filename and '_y' in filename and 'UW' in filename:
        y_mid_inflated = np.load('data\\{}grid_UW_y.npy'.format(grid_mid, inflation_factor))

    # Coarse grid UW data
    if grid_coarse in filename and '_phi' in filename and 'UW' in filename:
        phi_coarse_inflated = np.load('data\\{}grid_UW_phi.npy'.format(grid_coarse, inflation_factor))
    elif grid_coarse in filename and '_x' in filename and 'UW' in filename:
        x_coarse_inflated = np.load('data\\{}grid_UW_x.npy'.format(grid_coarse, inflation_factor))
    elif grid_coarse in filename and '_y' in filename and 'UW' in filename:
        y_coarse_inflated = np.load('data\\{}grid_UW_y.npy'.format(grid_coarse, inflation_factor))


# Compute spacing
spacing_fine = length_x/(len(x_fine) + 1)
spacing_mid = length_x/(len(x_mid) + 1)
spacing_coarse = length_x/(len(x_coarse) + 1)

# Create interpolations
f_fine = RectBivariateSpline(x_fine, y_fine, phi_fine)
f_mid = RectBivariateSpline(x_mid, y_mid, phi_mid)
f_coarse = RectBivariateSpline(x_coarse, y_coarse, phi_coarse)
# Compute error norms
e_coarse = norm_l2(f_fine(x_coarse, y_coarse), phi_coarse)
e_mid = norm_l2((f_fine(x_mid, y_mid)), phi_mid)
# Compute order of convergence
order = order_conv(e_coarse, e_mid, spacing_coarse, spacing_mid)

# Create interpolations for inflated mesh
f_fine_inflated = RectBivariateSpline(x_fine_inflated, y_fine_inflated, phi_fine_inflated)
# Compute error norms for inflated mesh
e_coarse_fine_interp = f_fine_inflated(x_coarse_inflated, y_coarse_inflated)
e_coarse_inflated = norm_l2(f_fine_inflated(x_coarse_inflated, y_coarse_inflated), phi_coarse_inflated)
e_mid_fine_interp = f_fine_inflated(x_mid_inflated, y_mid_inflated)
e_mid_inflated = norm_l2((f_fine_inflated(x_mid_inflated, y_mid_inflated)), phi_mid_inflated)
# Compute order of convergence for inflated mesh
order_inflated = order_conv(e_coarse_inflated, e_mid_inflated, spacing_coarse, spacing_mid)

# Print order of convergence values
print('CD Scheme order of convergence: {}'.format(order))
print('UW Scheme order of convergence: {}'.format(order_inflated))

# False Diffusion Plot
fig1 = plt.figure()
files = os.listdir('data\\falseDiffusion\\')
for filename in sorted(files):
    z = np.load('data\\falseDiffusion\\{}'.format(filename))
    diag_reverse = np.fliplr(z).diagonal()
    diag = np.flip(diag_reverse)
    spacing = length_x / (len(diag) + 1)
    x_diag = np.linspace(0, np.sqrt(length_x**2 + length_y**2), len(diag))
    plt.plot(x_diag, diag, label='Upwind {}$\\times${}'.format(len(diag), len(diag)))
exact = np.concatenate([100*np.ones(int(np.floor(len(x_diag)/2))), np.zeros(int(np.ceil(len(x_diag)/2)))])
# x_diag = np.linspace(0, np.sqrt(length_x**2 + length_y**2), len(exact))
plt.plot(x_diag, exact, label='Exact')
plt.xlabel('Distance Along Diagonal [m]')
plt.ylabel('$\phi$')
plt.title('False Diffusion and Grid Size')
plt.legend()
plt.tight_layout()
plt.savefig('plots\\fDiff.pdf')
plt.show()
