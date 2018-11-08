import numpy as np
import matplotlib.pyplot as plt
from utilities import *
#%%
# Asensio 2002 experiment with simulated wind
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
N_sim = 50
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
#U_sim, B_sim = getWildfireSimulations(parameters, V, N_sim)
# Save simulations
#np.save('../data/simulations/50Usim', U_sim)
#np.save('../data/simulations/50Bsim', B_sim)
# Load simulations
U_sim = np.load('../data/simulations/50Usim.npy')
B_sim = np.load('../data/simulations/50Bsim.npy')
#%%
# Plot a simulation
sim = 25
#plotSimulation(X, Y, U_sim, B_sim, V, sim)
plotSimulationHorizontal(X, Y, U_sim, B_sim, V, sim, save=True)
#%%
# Statistics of numerical simulations
tim = 0 # Select time of numerical simulation (0-9)
per_ = True
save_ = False
plotStatsCI(U_sim, tim, 'Temperature', per=per_, save=save_)
plotStatsCI(B_sim, tim, 'Fuel', per=per_, save=save_)
#%%
# Get fuel stats
meanB, stdB, lloB, lluB = getSimStats(B_sim, -1)
#%%
# Get initial fuel from any numerical simulation at time 0
total_B = (np.asarray(B_sim[0,0])).sum()
mean_end_B = (np.asarray(meanB)).sum()
burnt_per = (total_B - mean_end_B) / total_B * 100

plt.imshow(meanB, origin="lower", cmap=plt.cm.Oranges)
plt.colorbar()
print("Total initial fuel", total_B)
print("Mean fuel at end", mean_end_B)
print("Burnt %", burnt_per)

#%%
# Answer probabilities' questions
total_B = (np.asarray(B_sim[0,0]) > 0.2).sum()
mean_end_B = (np.asarray(meanB) <= 0.2).sum()
burnt_per = (total_B - mean_end_B) / total_B * 100

plt.imshow(meanB, origin="lower", cmap=plt.cm.Oranges)
plt.colorbar()
print("Total initial fuel", total_B)
print("Mean fuel at end", mean_end_B)
print("Burnt %", burnt_per)
#%%
# Answer probabilities' questions
per_fuel = np.zeros(N_sim)
for i in range(N_sim):
  #total_B = (np.asarray(B_sim[0,0]) > per).sum()
  #end_B = (np.asarray(B_sim[i,-1]) <= per).sum()
  total_B = (np.asarray(B_sim[0,0])).sum()
  end_B = (np.asarray(B_sim[i,-1])).sum()
  per_fuel[i] = (total_B - end_B) / total_B
#%%
per = 0.03
print((per_fuel > per).sum() / N_sim)
#%%
print((U_sim > 9).sum() / (50*10*128*128))
