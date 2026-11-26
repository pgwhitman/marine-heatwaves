import numpy as np
import matplotlib.pyplot as plt

from config import ModelConfig
from grid import Grid
from model import Model
from initial_conditions import initialise
from time_evolution import evolve
from plotting import plot_depth


#Set up configuration and grid
config = ModelConfig()
grid = Grid(config)

#Initialise Fields
state, b = initialise(grid, config)

#Initialise model
model = Model(config, b)

#Time integration
nsteps = int(config.t_end / config.dt)

# Evolve the solution
state = evolve(state, model, config)

# Plot final solution
plot_depth(grid, state)