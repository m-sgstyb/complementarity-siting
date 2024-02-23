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


## Data
The current version includes a siting and sizing optimisation for twelve potential wind and solar generation sites across Mexico. 
