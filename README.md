# Simulation of Dynamic Spatial Matching of Two Sided Queues

### Short description of important files:
Each notebook corresponds to one of the examples in the report:
- spatial.py: Driver file for running simulations with "spatial" weight matrix. Writes the total queue and cost paths along with input parameters in a pickle file.
- random_weights.py: Driver file for running simulations with "random" weight matrix. Writes the total queue and cost paths along with input parameters in a pickle file.
- utils.py: Contains the core code for simulating each path and for processing output data
- plots.ipynb: This and the other notebooks are used for plotting and analysing the output

### Requirements
- numpy
- cvxpy
- GLPK_MI solver
- gurobi
- pickle

### Running the code on PACE
```Shell
bash file_name.sh
```
Output will be written to a pickle file: sim_paths.pkl
