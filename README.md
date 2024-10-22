# Transfer Learning Gaussian Processes

Comparison of different transfer learning Gaussian process methods on synthetic data. This comparison was conducted for the paper _Transfer Learning Bayesian Optimization to Design Competitor DNA Molecules for Use in Diagnostic Assays_ [1].

## Installation

To clone this repo use the command:
    
     git clone https://github.com/RSedgwick/TLGPs.git

The code is written in Python 3.6. To install the required packages, ensure conda is installed and run the following 
command in the root directory of the project:

     conda env create -f environment.yml 

## Overview 

In this repo, we run experiments to compare four different transfer learning methods:

1. Independent multioutput Gaussian process (MOGP) [2]
2. Average Gaussian process where all data is considered to be from the same surface (AvgGP)
3. Linear Model of Coregionalisation (LMC) [2]
4. Latent Variable Multioutput Gaussian Process (LVMOGP) [3]

We run experiments for all these methods on three different test function scenarios based on situations in which we
expect each model to perform well:
1. Unrelated test functions 
   1. if there is no negative transfer we would expect the MOGP, LMC and LVMOGP to perform similarly here
2. Linearly-related test functions
   1. We expect the LMC and LVMOGP to outperform the MOGP here
3. Non-linearly-related test functions
   1. We expect the LVMOGP to outperform the MOGP and LMC here
We expect the MOGP, LMC and LVMOGP to outperform the AvgGP on all test scenarios.

## Results

Below is a plot of the mean of the root mean squared error (RMSE) and negative log predictive density (NLPD) for each of the methods for three 
different test function scenarios for one seed. For each scenario, we have 10 new surfaces being learnt and 5 different random data sets. This 
plot appears in Figure 4 of Sedgwick et al. [1]

![image](analysis/plots/learning_curves_seed_2_mean_potrait.svg)

Below is a gif of the predictions of each of the models for one test scenario, 
where one random datapoint is added each time.

![image](analysis/plots/predictions_unrelated_two_observed_10_new_points_seed_1_dataseed_1.gif)

## Code Overview

- `models`
  - `initializations.py` - Contains the initialization functions for the different transfer learning methods
  - `lvmogp.py` - Contains the code for the latent variable multi-output Gaussian process model, adapted from the GPflow Bayesian GPLVM code [4]
  - `test_functions.py` - Code for generating the synthetic data
- `utils`
  - `utils.py` - Useful functions for initialising models, fitting them, getting performance metrics and saving results
  - `plotting_utils.py` - Useful functions for plotting functions and performance metrics
  - `analysis_utils.py` - Useful functions for analysing the results
- `notebooks`
  - `Comparing_LVMOGP_Prediction_Methods.ipynb` - Notebook for comparing different methods for prediction using the LVMOGP
  - `fitting_all_models.ipynb` - Notebook for fitting all the models and saving plots of the predictions
  - `lmc_fitting_and_intialisation.ipynb` - Notebook demonstrating the fitting of the LMC
  - `lmc_setting_W_and_kappa.ipynb` - Notebook demonstrating how the LMC can recreate the independent MOGP
- `analysis`
  - `plots` - various plots that have been generated
  - `model_comparison.ipynb` - Notebook comparing the RMSE and NLPD of the different models from the many learning curve runs
  - `plot_predictions.ipynb` - plot the predictions of the models for a given seed, data seed and number of training points
Also, plot the log marginal likelihood of the different initialisations at each number of training points for all models
  - `plot_predictions.py` - plot the predictions for all runs, to be made into gifs
  - `animating_plots.ipynb` - notebook for making gifs out of predictions
- `experiments`
  - `learning_curves.py` - this script is used for fitting each of the models, analysing the results and saving them to a file
  - The `.pbs` scripts can be used to run this many times for different seeds and number of training points on a cluster
 
## References

[1] [Sedgwick, Ruby and Goertz, John and Stevens, Molly and Misener, Ruth and van der Wilk, Mark. "Transfer Learning Bayesian Optimization to Design Competitor DNA Molecules for Use in Diagnostic Assays" (2023)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/bit.28854)
[2] [Álvarez, Mauricio A., Lorenzo Rosasco, and Neil D. Lawrence (June 2012). “Kernels for Vector-Valued Functions: A Review”. doi: 10.1561/2200000036.](https://arxiv.org/abs/1106.6251)
[3] [Dai, Zhenwen, Mauricio Álvarez, and Neil Lawrence. "Efficient modeling of latent information in supervised learning using gaussian processes." Advances in Neural Information Processing Systems 30 (2017).](https://arxiv.org/abs/1705.09862)
[4] [Matthews, Alexander G. et al. (Jan. 2017). “GPflow: a Gaussian process library using tensorflow”. In: The Journal of Machine Learning Research 18.1, pp. 1299–1304. issn: 1532-4435](https://jmlr.org/papers/volume18/16-537/16-537.pdf)
 
## How to Cite 
When using the code in this repository, please reference our journal paper:
```
@article{sedgwick_transfer_2023,
  title={Transfer Learning Bayesian Optimization for Competitor DNA Molecule Design for Use in Diagnostic Assays},
  author={Sedgwick, Ruby and Goertz, John and Stevens, Molly and Misener, Ruth and van der Wilk, Mark},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={}
}
```
## Acknowledgements
This work was supported by the [UKRI CDT in AI for Healthcare](https://ai4health.io/) Grant No. EP/S023283/1 
