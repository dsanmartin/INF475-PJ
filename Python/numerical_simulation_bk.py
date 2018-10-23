import numpy as np
import matplotlib.pyplot as plt
from utilities import *
#%%
# Asensio 2002 experiment
M, N = 128, 128 # Space resolution
L = 1440 # Timesteps (10 per simulated values)
dt = 1e-2 # dt
xa, xb = 0, 90 # x domain limit
ya, yb = 0, 90 # y domain limit
x = np.linspace(xa, xb, N) # x domain
y = np.linspace(ya, yb, M) # y domain
t = np.linspace(0, dt*L, L) # t domain

# Parameters of wildfire physical model
kappa = 1e-1 # diffusion coefficient
epsilon = 3e-1 # inverse of activation energy
upc = 3 # u phase change
q = 1 # reaction heat
alpha = 1e-3 # natural convection
gamma = .5  # wind effect coefficient

# Temperature initial condition
u0 = lambda x, y: 6 * G(x-45, y-45, 20)#np.exp(-5e-2*((x-45)**2 + (y-45)**2)) 

# Fuel initial condition. Random uniform
np.random.seed(666)
b0 = lambda x, y: np.round(np.random.uniform(size=(x.shape)), decimals=2)

# Wind effect from simulated data (SARIMA models)
N_sim = 10
V = getSimulatedWind(N_sim, L, gamma)
#%%
# Meshes for initial condition plots
X, Y = np.meshgrid(x, y)

# Plot initial conditions
plotInitialConditions(X, Y, u0, b0)

# Parameters for the model
parameters = {
    'u0': u0, 
    'beta0': b0,
    'v': V,
    'kappa': kappa, 
    'epsilon': epsilon, 
    'upc': upc, 
    'q': q, 
    'alpha': alpha, 
    'x': x, 
    'y': y,
    't': t,
    'sparse': True,
    'show': False,
    'complete': False
}

#%%
# Generate wildfires simulations
Usim, Bsim = getWildfireSimulations(parameters, V, N_sim)
#%%
sim = 3
plotSimulation(X, Y, Usim, Bsim, V, sim)
#%%
tim = -1 # Select time of numerical simulation (0-9)
plotStatsCI(Usim, tim, 'Temperature')
plotStatsCI(Bsim, tim, 'Fuel')
#%%
# Compute mean and std of simulations
meanU = sum(Usim[i, tim] for i in range(10)) / 10
stdU = sum( (Usim[i, tim] - meanU) ** 2 for i in range(10)) / 10
lupU = meanU + 1.96*stdU / (10 ** 0.5)
lloU = meanU - 1.96*stdU / (10 ** 0.5)

meanB = sum(Bsim[i, tim] for i in range(10)) / 10
stdB = sum( (Bsim[i, tim] - meanB) ** 2 for i in range(10)) / 10
lupB = meanB + 1.96*stdB / (10 ** 0.5)
lloB = meanB - 1.96*stdB / (10 ** 0.5)

## Plot mean and confidence intervals
## Temperature
#plt.figure(figsize=(12, 8))
## Fuel
#plt.subplot(1, 3, 1)
#lloup = plt.imshow(lloU, origin="lower", cmap=plt.cm.jet, extent=[0, 90, 0, 90])
#plt.colorbar(lloup, fraction=0.046, pad=0.04)
#plt.title("Lower")
#plt.subplot(1, 3, 2)
#temp = plt.imshow(meanU, origin="lower", extent=[0, 90, 0, 90], cmap=plt.cm.jet)
#plt.colorbar(temp, fraction=0.046, pad=0.04)
#plt.title("Mean")
#plt.subplot(1, 3, 3)
#lupup = plt.imshow(lupU, origin="lower", cmap=plt.cm.jet, extent=[0, 90, 0, 90])
#plt.colorbar(lupup, fraction=0.046, pad=0.04)
#plt.title("Upper")
#plt.tight_layout()
#plt.show()
#
## Fuel
#plt.figure(figsize=(12, 8))
#plt.subplot(1, 3, 1)
#llobp = plt.imshow(lloB, origin="lower", cmap=plt.cm.Oranges, extent=[0, 90, 0, 90])
#plt.colorbar(llobp, fraction=0.046, pad=0.04)
#plt.title("Lower")
#plt.subplot(1, 3, 2)
#fuel = plt.imshow(meanB, origin="lower", extent=[0, 90, 0, 90], cmap=plt.cm.Oranges)
#plt.colorbar(fuel, fraction=0.046, pad=0.04)
#plt.title("Mean")
#plt.subplot(1, 3, 3)
#lupbp = plt.imshow(lupB, origin="lower", cmap=plt.cm.Oranges, extent=[0, 90, 0, 90])
#plt.colorbar(lupbp, fraction=0.046, pad=0.04)
#plt.title("Upper")
#plt.tight_layout()
#plt.show()

#%%
perro = np.zeros((100, M, N))
for j in range(100):
  sumU = sum(Usim[i, tim] for i in range(10)) 
  perro[j] = sumU
  
#%%
gato = Usim[:, tim].reshape((10, 10, M, N))
burro = np.sum(gato,axis=1)
mu = 10
n = 300
sums = 1000
suma = np.zeros(sums)
for i in range(sums):
  suma[i]= np.sum(np.random.normal(mu,1,n))/n
  
  
#plt.show()
#meanU[meanU < 1e-14] = 0
#lup[lup < 1e-14] = 0
#ldo[ldo < 1e-14] = 0
#plt.contour(X, Y, meanU, alpha=0.75)
#llup = plt.contour(X, Y, lup, 10, alpha=0.85)#, levels=1)
#plt.colorbar(llup)
#plt.clabel(llup, fontsize=9, inline=1)
#plt.contourf(X, Y, ldo, alpha=0.25)
#plt.colorbar()

#%%

#X, Y = np.meshgrid(x, y)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#cset = ax.contourf(X, Y, meanU, offset=50, cmap=cm.jet, alpha=0.75)
#cset = ax.contourf(X, Y, lup, offset=0, cmap=cm.jet, alpha=0.75)
##cset = ax.contourf(X, Y, ldo, offset=80, cmap=cm.jet, alpha=0.75)
#
##cset = ax.plot_surface(X, Y, meanU, cmap=cm.jet)#, offset=50, cmap=cm.jet, alpha=0.75)
##cset = ax.plot_surface(X, Y, 10 + lup, cmap=cm.jet, alpha=0.25)
##cset = ax.plot_surface(X, Y, ldo, cmap=cm.jet, alpha=0.25)
#
#ax.set_xlabel('X')
#ax.set_xlim(0, 90)
#ax.set_ylabel('Y')
#ax.set_ylim(0, 90)
#ax.set_zlabel('Z')
#ax.set_zlim(0, 100)
#ax.view_init(azim=260, elev=30)
#plt.show()