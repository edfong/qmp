# QMP: Quantile Martingale Posteriors
This repository contains the code for the illustrations in the preprint "" by Edwin Fong & Andrew Yiu. 

The run scripts 


## Installation
The `qmp` package can be installed by running the following in the main folder:
```
pip3 install ./
```
or the following if you'd like to make edits to the package:
```
 pip3 install -e ./
```
We recommend doing so in a clean virtual environment if reproducing the experimental results is of interest.

## Running Experiments
All experiment can be found in `run_scripts`.
```
python3 run_sim.py
```
The R scripts can be run in RStudio or terminal.

Outputs from the experiments are stored in `plots/plot_data', and all plots can be produced by the Jupyter notebooks in `plots'. 

## Data
The simulated data is produced in the run scripts. The cyclone dataset can be found here: https://myweb.fsu.edu/jelsner/temp/Data.html.


## Acknowledgements
We thank Hyoin An for providing the code for the DQP method which we used for our experiments. The R scripts within the `run_scripts/dqp_cyclone' folder are modified versions of the `application' version here: https://github.com/hyoin-an/DQP.