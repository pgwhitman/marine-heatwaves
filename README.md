Code to solve the shallow water equations to study marine heatwaves

Change to run different sims:

initial_conditions.py : Adjust this to run main.py for different I.C.s  
main.py : Run this  
config.py : Simulation and non-state variables defined

Other files:

grid.py : creates grid structures  
model.py : The update which occurs at a timestep  
operators.py : Defines numerical operators e.g. grad  
plotting.py : Plotting routines  
state.py : Defines dataclass State which contains complete information at any time  
time_evolution.py : Functions associated with evolving in time  

requirement.txt contains packages that must be installed


Potential updates:  
-Coriolis parameter is currently constant - can change this  
-Currently periodic B.C.s If coriolis changes, won't work so might need to change to fixed for 2 sides  
-Create better plotting routines  