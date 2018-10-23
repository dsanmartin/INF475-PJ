"""
Utilities function for numerical simulation.
For wildfire numerical simulation, we use this implementation
https://github.com/dsanmartin/ngen-kutral
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wildfire.fire import Fire
from wildfire import plots as p

# Simple Gaussian basis assuming sigma_x = sigma_y
G = lambda x, y, s: np.exp(-1/s * (x**2 + y**2))
# \partial G/ \partial x
Gx = lambda x, y, s: -2/s * x * G(x, y, s) 
# \partial G/ \partial y
Gy = lambda x, y, s: -2/s * y * G(x, y, s) 

# Create wind using speed and direction info
def createWindVelocity(speed, direction):
  """
  Create vector field for wind velocity
  (X, Y) = (r \cos(\theta), r \sin(\theta))
  speed (r): numpy array 
  direction (\theta): numpy array
  """
  angle = np.radians((direction + 180) % 360) # degrees to radians 
  X = np.multiply(speed, np.cos(angle))
  Y = np.multiply(speed, np.sin(angle))
  return X, Y

# Create wind input from simulated data
def getSimulatedWind(N_sim, timesteps, gamma):
  """
  N_sim: number of simulations (int)
  timesteps: Numerical simulation timesteps
  gamma: physical model coefficient of wind
  """
  V = np.zeros((N_sim, timesteps, 2)) # Simulated wind
  for i in range(1, N_sim+1):  
    speed = pd.read_csv('../data/simulated/speed/' + str(i) + '.csv').iloc[:,[1]].values
    direction = pd.read_csv('../data/simulated/direction/' + str(i) + '.csv').iloc[:,[1]].values
    #wX, wY = createWindVelocity(gamma * speed, gamma * 360*direction/(np.max(direction)-np.min(direction)))
    wX, wY = createWindVelocity(speed, direction)
    # Since we have a 10 minutes frequency, we repeat 10 times the information for each minute
    # We avoid interpolate for the data smooth involved
    for j in range(144):
      V[i-1, 10*j:10*(j+1), :] = np.array([[gamma * float(wX[j]), gamma * float(wY[j])]]*10)
  return V

# Generate wildfires simulations
def getWildfireSimulations(parameters, V, N_sim):
  """
  parameters: dictionary with physical model parameters
  V: numpy array with simulated wind
  N_sim: number of numerical simulations
  """
  M, N = len(parameters['x']), len(parameters['y'])
  Usim = np.zeros((N_sim, 10, M, N)) 
  Bsim = np.zeros((N_sim, 10, M, N))
  
  for i in range(N_sim):
    parameters['v'] = V[i]
    ct = Fire(parameters)
    U, B = ct.solvePDEData('fd', 'rk4')
    # We keep with 10 samples to analyze results
    Usim[i] = U[::144,:,:]
    Bsim[i] = B[::144,:,:]
  
  return Usim, Bsim

# Initial condition of numerical simulation
def plotInitialConditions(X, Y, u0, b0):
  """
  X, Y = numpy meshgrid
  u0 = initial temperature lambda
  b0 = initial temperature lambda
  """
  p.plotIC(X, Y, u0, b0, W=None, T=None, top=None)
  
# Plot a simulation
def plotSimulation(X, Y, Usim, Bsim, V, sim):
  """
  X, Y = numpy meshgrid with domain
  Usim, Bsim = numpy arrays with temperature and fuel approximations
  V: simulated wind
  sim: number of simulation to plot
  """
  pU = Usim[sim]
  pB = Bsim[sim]
  pV = V[sim,::144,:]
  for i in range(len(pU)):
    f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 4))
    temp = ax1.imshow(pU[i], origin="lower", alpha=0.7, extent=[X[0, 0], X[0, -1], 
                  Y[0, 0], Y[-1, 0]], cmap=plt.cm.jet)
    plt.colorbar(temp, fraction=0.046, pad=0.04, ax=ax1)    
    ax1.quiver(X[::8, ::8], Y[::8, ::8], pV[i, 0]*np.ones((16, 16)), pV[i, 1]*np.ones((16, 16)))
    fuel = ax2.imshow(pB[i], origin="lower", extent=[X[0, 0], X[0, -1], 
                  Y[0, 0], Y[-1, 0]], cmap=plt.cm.Oranges)
    plt.colorbar(fuel, fraction=0.046, pad=0.04, ax=ax2)
    plt.show()
    
# Get simulations stats
def getSimStats(Asim, tim, cl=95):
  """
  Asim: approximation numpy array
  tim: time to analize
  cl: convidence level
  """
  z = {80: 1.28, 90: 1.645, 95: 1.96, 98: 2.33, 99: 2.58}
  n = len(Asim)
  mean = sum(Asim[i, tim] for i in range(n)) / n
  std = sum( (Asim[i, tim] - mean) ** 2 for i in range(n)) / n
  lup = mean + z[cl]*std / ((n-1) ** 0.5)
  llo = mean - z[cl]*std / ((n-1) ** 0.5)
  return mean, std, llo, lup

# Plot mean and confidence intervals of approximation
def plotStatsCI(A, tim, approx, cl=95):
  """
  A: approximation numpy array
  tim: time to analize
  approx: Approximation type 'Temperature' or 'Fuel'
  """
  if approx == 'Temperature':
    cmap_ = plt.cm.jet
  elif approx == 'Fuel':
    cmap_ = plt.cm.Oranges
  else:
    print("Error in approximation type")
    return
  
  f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(12, 8))
  mean, std, llo, lup = getSimStats(A, tim, cl)
  llop = ax1.imshow(llo, origin="lower", cmap=cmap_, extent=[0, 90, 0, 90])
  plt.colorbar(llop, fraction=0.046, pad=0.04, ax=ax1)
  ax1.set_title("Lower")
  meanp = ax2.imshow(mean, origin="lower", extent=[0, 90, 0, 90], cmap=cmap_)
  plt.colorbar(meanp, fraction=0.046, pad=0.04, ax=ax2)
  ax2.set_title(approx + " Mean")
  lupp = plt.imshow(lup, origin="lower", cmap=cmap_, extent=[0, 90, 0, 90])
  plt.colorbar(lupp, fraction=0.046, pad=0.04, ax=ax3)
  ax3.set_title("Upper")
  plt.show()