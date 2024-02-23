# Spatio-temporal complementarity metrics optimisation for siting wind and PV plants
The current version includes an example for twelve sites across Mexico.


## To run the code in a local repository 
Go to the path where you want to save the repository:
```
(base) base_path % cd your_path/your_folder_name
(base) .../your_folder_name % git clone https://github.com/m-sgstyb/complementarity-siting
```
### Dependencies
This model requires the following dependencies, which can be installed individually:
1. [CVXPY](https://github.com/cvxpy/cvxpy/tree/master)
2. [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
3. [NumPy](https://numpy.org/install/)
4. [matplotlib](https://matplotlib.org/stable/users/installing/index.html) To plot some of the results

Alternatively, an environment can be created through a conda package manager:
```
(base) .../your_folder_name % cd complementarity-siting
(base) .../your_folder_name/ % conda env create -f environment.yaml 
```
This includes the required dependencies, as well as the installation of the MOSEK optimisation suite, which is optional (see below), and the spyder IDE.

### Solver
The current optimisation may be run with open source solvers, such as ECOS or SCS.

The results shown have been obtained running the model using the commercial MOSEK solver, through an academic license. Note that solver choice may yield a different result, in particular for GEN optimisation. Solver selection has not been found to create different results for the other metrics. An academic license for MOSEK can be obtained [here]([https://www.mosek.com](https://www.mosek.com/products/academic-licenses/)), which should reproduce the same results.

## Running the model
### Data
The current version includes a siting and sizing optimisation for twelve potential wind and solar generation sites across Mexico. To run the model for this test data, follow the instructions below.

### Run optimisation
The optimisation will run on main.py
1. Define the sizing parameter (alpha) to be used by changing line 32 and the complementarity metric to be used on line 33. Indicating these values will ensure that the demand constraint for the sizing of the overall system is met, and that the files are saved with the appropriate names. For example, for the system sized to produce the same amount of energy as total annual demand, and optimised for Peak Residual Demand (PRD):
```
al = 1
metric = "PRD"
```

2. Go to lines 126 - 131, and uncomment only the objective for desired metric that was input above. To optimise PRD, this should be:
```
objective = cp.Minimize(peak_res_demand)
# objective = cp.Minimize(avg_res_demand)
# objective = cp.Maximize(total_generation)
# objective = cp.Minimize(lolp)
# objective = cp.Minimize(var)
# objective = cp.Minimize(max_var)
```

3. If using an academic license for MOSEK, run main.py; if not, comment line 137 and uncomment line 138:
```
# result = prob.solve(solver = cp.MOSEK, warm_start=True, qcp=False, verbose=False)
result = prob.solve(solver = cp.SCS, warm_start=True, max_iters=1000, verbose=False)
```

Steps 1. and 2. can be repeated for the six metrics currently in the code. Additional metrics can be added if desired.

### Results and plotting
All results are saved in the results folder after each run. .csv files will be created for the installed capacities, as well as energy flows.

The script for visualizations.py includes functions that plot some of the results; examples of how to plot and save the figures can be found at the bottom of this script.
