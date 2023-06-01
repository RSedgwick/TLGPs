
import pathlib as pl
import os
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
from utils.plotting_utils import plot_all_models_from_hyperparameters, plot_lvmogp_latent_variables
from utils.analysis_utils import load_hyperparameters, fit_model_from_hyperparameters

full_width = 7.5
halfwidth = 3.5

surface_type = 'linear_relation'
test_type = 'two_observed_10_new'
n_new_points = 1

plot_path = pl.Path().home()/'TLGP_plots'

for surface_type in ['unrelated', 'linear_relation', 'non-linear_relation']:
       for n_new_points in range(2, 30):
              for seed in [1, 2, 3]:
                     for dataseed in range(6):
                            path = pl.Path.home() / f'Transfer_Learning_GP_Results/hyperparameters/hyperparameters_{surface_type}_{test_type}_{n_new_points}_points_seed_{seed}_dataseed_{dataseed}.pkl'
                            if os.path.exists(path):
                                   hyperparams = load_hyperparameters(path)
                                   hyperparams['final_lml'] = [hyperparams['lmls'].to_numpy()[i][-1] for i in range(len(hyperparams))]

                                   gps = {model_name: None for model_name in ['mo_indi', 'avg', 'lmc', 'lvmogp']}
                                   for model_name in ['mo_indi', 'avg', 'lmc', 'lvmogp']:
                                          print(model_name)
                                          mod_df = hyperparams[hyperparams['model'] == model_name]
                                          gps[model_name] = fit_model_from_hyperparameters(mod_df)
                                   save_path = plot_path / f'predictions_{surface_type}_{test_type}_{n_new_points}_points_seed_{seed}_dataseed_{dataseed}'
                                   plot_all_models_from_hyperparameters(gps, hyperparams, plot_new_ys=False,
                                                                        save_fig=True, save_path=save_path)
                                   plot_lvmogp_latent_variables(gps['lvmogp'], save=True, path=plot_path,
                                                                file_name=f'latent_variables_{surface_type}_{test_type}_{n_new_points}_points_seed_{seed}_dataseed_{dataseed}')
                                   plt.close()

