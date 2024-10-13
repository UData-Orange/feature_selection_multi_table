from khiops import core as kh
from math import *
from os import path
from tqdm import tqdm 
import timeit

def get_key(dictionary, val):
        """Get the key of a value in a python dictionary

        :param dictionary: Python dictionary.
        :type dictionary: dict
        :param val: Value to test.
        :return: The key associated to the value

        """
        for key, value in dictionary.items():
            if value == val:
                return key
        return -1
       
def add_noise(dictionary_domain, number_noise):
    """Add noise variable into secondaries tables. 

    :param dictionary_domain: Khiops dictionary domain
    :param number_noise: Number of noise variable to add per table
    :type number_noise: int
    :return: Khiops dictionary domain with noise
    """
    #Initialize
    index = 0
    noise_variable = []
    for dictionary in dictionary_domain.dictionaries:
        if not dictionary.root :
            number_add_noise = 0
            for i in range(number_noise):
                # Add numerical noise variable
                if number_add_noise < number_noise :
                    variable = kh.Variable()
                    variable.name = 'N_' + str(index)
                    noise_variable.append(variable.name)
                    variable.type = 'Numerical'
                    variable.used = True
                    variable.rule = 'Sum(Random(),'+str(i)+')'
                    dictionary_domain.get_dictionary(dictionary.name).add_variable(variable)
                    noise_variable.append(variable.name)
                    number_add_noise+=1
                # Add categorical noise variable
                if number_add_noise < number_noise :
                    variable = kh.Variable()
                    variable.name = 'C_' + str(index)
                    noise_variable.append(variable.name)
                    variable.type = 'Categorical'
                    variable.used = True
                    variable.rule = "Concat(\"V_\",FormatNumerical(Round(Product("+str(2**(i+1))+",Random())),0,0))"
                    dictionary_domain.get_dictionary(dictionary.name).add_variable(variable)
                    noise_variable.append(variable.name)
                    number_add_noise+=1
                index += 1
    return dictionary_domain, noise_variable

def remove_variable(dictionary_domain, variable_name):
    """Set to unused a variable in khiops dictionary domain

    :param dictionary_domain: Khiops dictionary domain
    :param variable_name: Name of the variable to remove
    :type variable_name: str
    :return: Train and test AUC values
    :rtype: list
    """
    for dict in dictionary_domain.dictionaries:
        for variable in dict.variables:
            if variable_name == variable.name:
                dictionary_domain.get_dictionary(dict.name).get_variable(variable_name).used = False
                print("var to remove found : " + str(variable_name))
    print("var removed : " + str(variable_name))
    return dictionary_domain

def filter_variable(dictionary_domain, 
                    importance_list, 
                    main_dictionary_name, 
                    data_table_path, 
                    target, 
                    additional_data_tables, 
                    result_dir, 
                    agregat_number, 
                    fold=0):
    """Filter variables by increasing order of importance and compute metric AUC

    :param dictionary_domain: Khiops dictionary domain.
    :param importance_list: List of variable importance by order.
    :type importance_list: list
    :param main_dictionary_name: Name of the dictionary to be analyzed.
    :type main_dictionary_name: str
    :param data_table_path: Path of the data table file.
    :type data_table_path: str
    :param target: Name of the target variable.
    :type target: str
    :param additional_data_tables: A dictionary containing the data paths and file paths for a multi-table dictionary file.
    :type additional_data_tables: dict
    :param result_dir: Path of the results directory.
    :type result_dir: str
    :param agregat_number: Maximum number of variables to cosntruct.
    :type agregat_number: int
    :param fold: Number of fold for Cross-validation, defaults to 0
    :type fold: int, optional
    :return: Train and tes metrics
    :rtype: list
    """
    # Initialize list results
    train_number = len(importance_list)+1
    train_auc_list = []
    test_auc_list = []
    sd_train_auc_list = []
    sd_test_auc_list = []
    for i in tqdm(range(train_number)):
        train_auc, test_auc, sd_train_auc, sd_test_auc = train_predictor_with_cross_validation(dictionary_domain,
                                                                                main_dictionary_name,
                                                                                data_table_path,
                                                                                target,
                                                                                result_dir,
                                                                                additional_data_tables,
                                                                                agregat_number,
                                                                                fold,
                                                                                i
                                                                                )
        train_auc_list.append(train_auc)
        test_auc_list.append(test_auc)
        sd_train_auc_list.append(sd_train_auc)
        sd_test_auc_list.append(sd_test_auc)
        
        # remove the variable with the lower importance
        if i!=train_number-1:
            dictionary_domain = remove_variable(dictionary_domain, importance_list[i][1])
            #print(dictionary_domain)
        
    return train_auc_list, test_auc_list, sd_train_auc_list, sd_test_auc_list

def filter_primitives(dictionary_domain, 
                    importance_list, 
                    main_dictionary_name, 
                    data_table_path, 
                    target, 
                    additional_data_tables, 
                    result_dir, 
                    agregat_number, 
                    construction_rule=kh.all_construction_rules,
                    fold=5):
    """Filter primitive by increasing order of importance and compute metric AUC

    :param dictionary_domain: Khiops dictionary domain.
    :param importance_list: List of primitive importance by order.
    :type importance_list: list
    :param main_dictionary_name: Name of the dictionary to be analyzed.
    :type main_dictionary_name: str
    :param data_table_path: Path of the data table file.
    :type data_table_path: str
    :param target: Name of the target variable.
    :type target: str
    :param additional_data_tables: A dictionary containing the data paths and file paths for a multi-table dictionary file.
    :type additional_data_tables: dict
    :param result_dir: Path of the results directory.
    :type result_dir: str
    :param agregat_number: Maximum number of variables to cosntruct.
    :type agregat_number: int
    :param construction_rule: list of construction rule, defaults to kh.all_construction_rules
    :type construction_rule: list, optional
    :param fold: Number of fold for Cross-validation, defaults to 0
    :type fold: int, optional
    :return: Train and tes metrics
    :rtype: list
    """
    # Initialize list results
    train_number = len(importance_list)+1
    train_auc_list = []
    test_auc_list = []
    sd_train_auc_list = []
    sd_test_auc_list = []
    
    for i in tqdm(range(train_number)):
        train_auc, test_auc, sd_train_auc, sd_test_auc = train_predictor_with_cross_validation(dictionary_domain,
                                                                                main_dictionary_name,
                                                                                data_table_path,
                                                                                target,
                                                                                result_dir,
                                                                                additional_data_tables,
                                                                                agregat_number,
                                                                                construction_rules=construction_rule,
                                                                                fold=fold
                                                                                )
        train_auc_list.append(train_auc)
        test_auc_list.append(test_auc)
        sd_train_auc_list.append(sd_train_auc)
        sd_test_auc_list.append(sd_test_auc)
        
        # remove the variable with the lower importance
        if i!=train_number-1:
            print("here")
            print(importance_list[i][0])
            construction_rule.remove(importance_list[i][0])
        
    return train_auc_list, test_auc_list, sd_train_auc_list, sd_test_auc_list

def train_predictor_without_cv(dictionary_domain,
        dictionary_name,
        data_table_path,
        target,
        result_directory,
        additional_table,
        number_agregat,
        construction_rules = kh.all_construction_rules,
        selection_variable = "",
        selection_value = "",
        results_prefix = "",
        max_trees = 0):
    """Train predictor on the dictionary using Khiops without cross validation.

    :param dictionary_domain: Khiops dictionary domain
    :param dictionary_name: Name of the dictionary to be analyzed.
    :type dictionary_name: str
    :param data_table_path: Path of the data table file.
    :type data_table_path: str
    :param target: Name of the target variable.
    :type target: str
    :param result_directory: Path of the results directory
    :type result_directory: str
    :param additional_table: A dictionary containing the data paths and file paths for a multi-table dictionary file.
    :type additional_table: dict
    :param number_agregat: maximum number of variables to construct
    :type number_agregat: int
    :param construction_rules: _Allowed rules for the automatic variable construction, defaults to kh.all_construction_rules., defaults to kh.all_construction_rules
    :type construction_rules: list, optional
    :param selection_variable: It trains with only the records such that the value of selection_variable is equal to selection_value, defaults to ""
    :type selection_variable: str, optional
    :param selection_value: See selection_variable option above, defaults to ""
    :type selection_value: str, optional
    :param results_prefix: Prefix of the result files, defaults to ""
    :type results_prefix: str, optional
    :param max_trees: Maximum number of trees to construct, defaults to 0
    :type max_trees: int, optional
    :return: train and test AUC
    :rtype: float
    """
    # Trainning
    json_result_file_path,_ = kh.train_predictor(
        dictionary_domain,
        dictionary_name,
        data_table_path,
        target,
        result_directory,
        additional_data_tables = additional_table,
        construction_rules = construction_rules,
        selection_variable = selection_variable,
        selection_value = selection_value,
        results_prefix = results_prefix + "_" + str(number_agregat),
        max_constructed_variables = number_agregat,
        max_trees = max_trees
        )
    
    # Analyse trainning step performance
    results = kh.read_analysis_results_file(json_result_file_path)
    # Add train and test AUC performance in train and test AUC list
    train_auc = results.train_evaluation_report.get_snb_performance().auc
    test_auc  = results.test_evaluation_report.get_snb_performance().auc
    return train_auc, test_auc

def train_predictor_with_cross_validation(dictionary_domain,
        dictionary_name,
        data_table_path,
        target,
        result_directory,
        additional_table,
        number_agregat,
        fold,
        i=0,
        construction_rules = kh.all_construction_rules,
        results_prefix = "",
        max_trees = 0,
        global_table = None,
        data_name = "", 
        type = ""):
    """Train predictor on the dictionary using Khiops with cross validation.

    :param dictionary_domain: Khiops dictionary domain
    :param dictionary_name: Name of the dictionary to be analyzed.
    :type dictionary_name: str
    :param data_table_path: Path of the data table file.
    :type data_table_path: str
    :param target: Name of the target variable.
    :type target: str
    :param result_directory: Path of the results directory
    :type result_directory: str
    :param additional_table: A dictionary containing the data paths and file paths for a multi-table dictionary file.
    :type additional_table: dict
    :param number_agregat: maximum number of variables to construct
    :type number_agregat: int
    :param fold: number of fold for cross validation
    :type fold: int
    :param construction_rules: Allowed rules for the automatic variable construction, defaults to kh.all_construction_rules., defaults to kh.all_construction_rules
    :type construction_rules: list, optional
    :param results_prefix: Prefix of the result files, defaults to ""
    :type results_prefix: str, optional
    :param max_trees: Maximum number of trees to construct, defaults to 0
    :type max_trees: int, optional
    :return: train and test AUC
    :rtype: float
    """
    fold_dictionary_file_path = path.join(result_directory, dictionary_name + "_" + str(i) + "_" + "Folding.kdic")
    # Load the learning dictionary object
    dictionary = dictionary_domain.get_dictionary(dictionary_name)

    # Add a random fold index variable to the learning dictionary
    fold_number = fold
    fold_index_variable = kh.Variable()
    fold_index_variable.name = "FoldIndex"
    fold_index_variable.type = "Numerical"
    fold_index_variable.used = False
    fold_index_variable.rule = "Ceil(Product(" + str(fold_number) + ",  Random()))"
    if fold_index_variable.name not in str(dictionary.variables):
        dictionary.add_variable(fold_index_variable)

    # Add variables that indicate if the instance is in the train dataset:
    for fold_index in range(1, fold_number + 1):
        is_in_train_dataset_variable = kh.Variable()
        is_in_train_dataset_variable.name = "IsInTrainDataset" + str(fold_index)
        is_in_train_dataset_variable.type = "Numerical"
        is_in_train_dataset_variable.used = False
        is_in_train_dataset_variable.rule = "NEQ(FoldIndex, " + str(fold_index) + ")"
        if is_in_train_dataset_variable.name not in str(dictionary.variables):
            dictionary.add_variable(is_in_train_dataset_variable)

    dictionary_domain.export_khiops_dictionary_file(fold_dictionary_file_path)

    train_aucs = []
    test_aucs = []
    for fold_index in tqdm(range(1, fold_number + 1)):
        start = timeit.default_timer()
        # Train a model from the sub-dataset where IsInTrainDataset<k> is 1
        train_reports_path, modeling_dictionary_file_path = kh.train_predictor(
            dictionary_domain,
            dictionary_name,
            data_table_path,
            target,
            result_directory,
            additional_data_tables = additional_table,
            sample_percentage=100,
            max_constructed_variables = number_agregat,
            construction_rules = construction_rules,
            selection_variable="IsInTrainDataset" + str(fold_index),
            selection_value=1,
            max_trees=max_trees,
            results_prefix="Fold" + str(fold_index) + "_" + str(number_agregat) + "_" + results_prefix,
        )

        # Evaluate the resulting model in the subsets where IsInTrainDataset is 0
        test_evaluation_report_path = kh.evaluate_predictor(
            modeling_dictionary_file_path,
            dictionary_name,
            data_table_path,
            result_directory,
            sample_percentage=100,
            additional_data_tables=additional_table,
            selection_variable="IsInTrainDataset" + str(fold_index),
            selection_value=0,
            results_prefix="Fold" + str(fold_index) + "_" + str(number_agregat),
        )

        stop = timeit.default_timer()
        # Obtain the train AUC from the train report and the test AUC from the
        # evaluation report and print them
        train_results = kh.read_analysis_results_file(train_reports_path)
        test_evaluation_results = kh.read_analysis_results_file(
            test_evaluation_report_path
        )
        train_auc = train_results.train_evaluation_report.get_snb_performance().auc
        test_auc = test_evaluation_results.evaluation_report.get_snb_performance().auc

        # Store the train and test AUCs in arrays
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        if global_table is not None:
            row = [data_name, type, fold_index, train_auc, test_auc, stop-start, number_agregat]
            global_table.loc[len(global_table)] = row
    # Print the mean +- error aucs for both train and test
    mean_train_auc = sum(train_aucs) / fold_number
    squared_error_train_aucs = [(auc - mean_train_auc) ** 2 for auc in train_aucs]
    sd_train_auc = sqrt(sum(squared_error_train_aucs) / (fold_number - 1))
    
    mean_test_auc = sum(test_aucs) / fold_number
    squared_error_test_aucs = [(auc - mean_test_auc) ** 2 for auc in test_aucs]
    sd_test_auc = sqrt(sum(squared_error_test_aucs) / (fold_number - 1))
    
    return mean_train_auc, mean_test_auc, sd_train_auc, sd_test_auc

def maximum_exist(list_importance):
    max = list_importance[0][1]
    index = 0
    exist = False
    for i in range(1,len(list_importance)):
        if list_importance[i][1] > max:
            index = i
            max = list_importance[i][1]
            
    max_95 = 0.9*max
    index_95 = 0
    if index < len(list_importance)-1:
        exist = True
        for i in range(1,len(list_importance)):
            if list_importance[i][1] > max_95:
                index_95 = i
                break
        return exist, index_95, list_importance[index_95][1]
    return exist, index, max

def select_variable_primitive(dictionary_domain, 
                              variable_importance = [], 
                              primitive_importance = [],
                              variable_number_select = 0,
                              selection_type = "non_informative",
                              path_dictionary=""
                              ):
    """Select informative variable and primitive to keep.

    :param dictionary_domain: Khiops dictionary domain
    :param variable_importance: List of variable importance by order, defaults to []
    :type variable_importance: list, optional
    :param primitive_importance: List of primitive importance by order, defaults to []
    :type primitive_importance: list, optional
    :param variable_number_select: Number of informative variable to be selected, defaults to 0
    :type variable_number_select: int, optional
    :param selection_type: Type of selection, defaults to "non_informative"
    :type selection_type: str, optional
    :param path_dictionary: path of the new created dictionary, defaults to ""
    :type path_dictionary: str, optional
    :return: variable importance list with only selected variable and primitive importance list with only selected primitive.
    :rtype: list
    """
    # Select only important items
    if selection_type == "non_informative":
        # Select variable
        if variable_importance != []:
            while variable_importance[0][2] == 0:
                dictionary_domain.get_dictionary(variable_importance[0][0]).get_variable(variable_importance[0][1]).used=False
                variable_importance.pop(0)
        # Select Primitive
        if primitive_importance != []:
            while primitive_importance[0][1] == 0:
                primitive_importance.pop(0)
    
    # Select a given variable number 
    if variable_number_select != 0 and variable_number_select < len(variable_importance):
        variable_to_unused = variable_importance[0:len(variable_importance)-variable_number_select-1]
        variable_importance = variable_importance[len(variable_importance)-variable_number_select:-1]
        for variable in variable_to_unused:
            dictionary_domain.get_dictionary(variable[0]).get_variable(variable[1]).used=False
    
    # Create new dictionary 
    dictionary_domain.export_khiops_dictionary_file(path_dictionary + "/dictionary_with_selection.kdic")
    
    return variable_importance, primitive_importance, path_dictionary + "/dictionary_with_selection.kdic"
    