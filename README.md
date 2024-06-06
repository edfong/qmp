# QMP: Quantile Martingale Posteriors
This repository contains the code for the illustrations in the preprint "Bayesian Quantile Estimation and Regression with Martingale Posteriors" by Edwin Fong & Andrew Yiu which can be found [here](https://arxiv.org/abs/2406.03358). 

## Installation
The `qmp` package can be installed by running the following in the main folder:
```
pip3 install ./
```
or the following if you'd like to make edits to the package:
```
 pip3 install -e ./
```
We recommend doing so in a clean virtual environment.

## Running Experiments
All experiment can be found in `run_scripts`, and can be run for example with:
```
python3 run_sim.py
```
The R scripts can be run in RStudio or terminal.

Outputs from the experiments are stored in `plots/plot_data`, and all plots can be produced by the Jupyter notebooks in `plots`. 

## Data
The cyclone dataset (Elsner et al. [2008]) is the second dataset under "Lifetime Maximum Wind Speeds (Global)" here: https://myweb.fsu.edu/jelsner/temp/Data.html. To run the experiments, download the data file `globalTCmax4.txt` into the `data` folder.


## Acknowledgements
We thank Hyoin An for providing the code for the DQP method (An and MacEachern [2024]) which we used for our experiments. The R scripts within the `run_scripts/dqp_cyclone` folder are modified versions of the author's original code, which can be found in the `application` folder here: https://github.com/hyoin-an/DQP. 

## References
J. B. Elsner, J. P. Kossin, and T. H. Jagger. The increasing intensity of the strongest tropical cyclones. Nature,
455(7209):92–95, 2008

H. An and S. N. MacEachern. A process of dependent quantile pyramids. Journal of Nonparametric Statistics,
pages 1–25, 2024.
