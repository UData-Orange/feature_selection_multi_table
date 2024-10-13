# Import python packages
import os
from os import path
from khiops import core as kh
import warnings
from math import *
import pickle
import sys
import pandas as pd
# Ajouter le chemin du dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function.some_function import *
from function.class_UnivariateMultitableAnalysis import UnivariateMultitableAnalysis
from function.dataset import get_dataset

os.environ["KHIOPS_PROC_NUMBER"]='5'
warnings.filterwarnings("ignore")

# Create results ans output directories
results_dir = "output_khiops"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

output_dir = "results"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if __name__ == "__main__":
    # Set information
    rule = kh.all_construction_rules            # construction rules to used
    k = 50           # agregat number per variable
    
    for dataset in ['Accident_star'] : 
        # Set Synthetique data information
        commun_path, \
        dictionary_file_path, \
        data_table_path, \
        secondary_table_list, \
        Additional_data_tables, \
        main_dictionary_name, \
        target = get_dataset(dataset)
        # Create dictionary with 10 noise variable
        dictionary_domain_10,_ = add_noise(kh.read_dictionary_file(dictionary_file_path), 10)
        dictionary_domain_10.export_khiops_dictionary_file(commun_path + "/noisy_dictionary.kdic")
        dictionary_file_path_noise = commun_path + "/noisy_dictionary.kdic"
            
        # Dictionary with 10 noise variable + filtering
        # Create Analyse Variable object
        obj = UnivariateMultitableAnalysis(dictionary_file_path_noise,
                            main_dictionary_name,
                            data_table_path,
                            Additional_data_tables,
                            target,
                            count_effect_reduction=True,
                            max_constructed_variables_per_variable=k,
                            result_file=dataset,
                            results_dir=results_dir,
                            output_dir=output_dir,
                            )
        
        # Compute importance variable list
        obj.analyse_variable()
        print(obj.variable_importance_final)

