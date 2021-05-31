# Analysis of kinetic models for the Bacterial Flagellar Motor

Python scripts that process BFM traces and compares the results with different prescribed kinetic metamodels, currently Hill-Langmuir, speed-rate, and two-state model.

## Files

* [data_analysis.py](data_analysis.py) Loads experimental traces and creates summary statistics that can be used to compare with theoretical results through the function `DistanceLikelihood()`
* [integration_ME.py](integration_ME.py) Integrates the Master Equation for different kinetic models through the diagonalization of the rate matrix. It also includes routines to manipulate the resulting probability vectors, integrate Mean-Field approximations, and reproduce stochastic trajectories.
* [models.py](models.py) Properties of different models related them to the respective father metamodel e.g. relating the cathbond model to the generic two-state model. It also includes the routines to define, sample and evaluate the prior distributions used in the ABC.
* [abc_smc.py](abc_smc.py) Aproximate Bayesian Computation using Sequential Monte Carlo sampling for the models defined in [models.py](models.py) and using the distance function defined in [data_analysis.py](data_analysis.py)
* [plots.py](plots.py) Plots the results of the ABC-SMC.

### Requirements

The external library versions used are:

scipy==1.5.0
seaborn==0.10.1
matplotlib==3.2.2
numpy==1.18.5
tqdm==4.47.0
pandas==1.0.5
p_tqdm==1.3.3 - Only required to use parallel mapping in the ABC-SMC sampling

## Author

* **Ruben Perez-Carrasco** - [2piruben](https://github.com/2piruben)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

    