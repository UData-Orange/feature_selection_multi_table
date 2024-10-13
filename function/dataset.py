"""dataset.py
    
Store data information for multiple dictionary.
"""
import os
from os import path

def get_dataset(dataset_name):
    if dataset_name == "Accident_star":
        # Set accident data information
        commun_path = "..\\DATA\\Accident"
        dictionary_file_path = path.join(commun_path, "Accidents_etoile.kdic")
        data_table_path = path.join(commun_path, "Accidents.txt")
        vehicle_table_path = path.join(commun_path, "Vehicles.txt")
        user_table_path = path.join(commun_path, "Users.txt")
        place_table_path = path.join(commun_path, "Places.txt")
        main_dictionary_name = 'Accident'
        Additional_data_tables={
            main_dictionary_name + "`Place": place_table_path,
            main_dictionary_name + "`Vehicles": vehicle_table_path,
            main_dictionary_name + "`Users": user_table_path
        }
        target = 'Gravity'
        return commun_path, \
            dictionary_file_path, \
            data_table_path, \
            [vehicle_table_path, user_table_path, place_table_path], \
            Additional_data_tables, \
            main_dictionary_name, \
            target
