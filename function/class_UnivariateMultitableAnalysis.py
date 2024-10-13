from math import *
from tqdm import tqdm
from khiops import core as kh
from os import path
import re
from tqdm import tqdm
import os
import pandas as pd
import sys
# Ajouter le chemin du dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
import some_function
from some_function import get_key

# Dictionary to match TablePartiton with construction rule
Dict_matching_partition = {'TablePartitionCount' : 'TableCount', 
                 'TablePartitionCountDistinct' : 'TableCountDistinct',
                 'TablePartitionMode' : 'TableMode',
                 'TablePartitionModeAt' : 'TableModeAt',
                 'TablePartitionMean' : 'TableMean',
                 'TablePartitionStdDev' : 'TableStdDev',
                 'TablePartitionMedian' :'TableMedian',
                 'TablePartitionMin' : 'TableMin',
                 'TablePartitionMax' : 'TableMax',
                 'TablePartitionSum' : 'TableSum'}

# Dictionary to match primitive name with construction rule
Dict_matching_rule_name = {'Count' : 'TableCount', 
                 'CountDistinct' : 'TableCountDistinct',
                 'Mode' : 'TableMode',
                 'ModeAt' : 'TableModeAt',
                 'Mean' : 'TableMean',
                 'StdDev' : 'TableStdDev',
                 'Median' :'TableMedian',
                 'Min' : 'TableMin',
                 'Max' : 'TableMax',
                 'Sum' : 'TableSum',
                 'Date' : 'GetDate',
                 'Time' : 'GetTime',
                 }


class UnivariateMultitableAnalysis():
    """Estimates the importance of secondary native variables in multitable 
    data on the target variable using a univariate approach with or without 
    discretization.
    
    :param dictionary_file_path: Path of a Khiops dictionary file.
    :type dictionary_file_path: str
    :param dictionary_name: Name of the dictionary to be analyzed.
    :type dictionary_name: str
    :param data_table_path: Path of the data table file.
    :type data_table_path: str
    :param additional_data_tables: A dictionary containing the data paths and file paths for a multi-table dictionary file.
    :type additional_data_tables: dict
    :param target_variable: Name of the target variable.
    :type target_variable: str
    :param exploration_type: Parameter to be analyze, 'All' for both variable and primitive, 'Variable' or 'Primitive for only variable or primitive,  defaults to 'Variable'.
    :type exploration_type: str
    :param count_effect_reduction: State of discretisation, True is used, defaults to True.
    :type count_effect_reduction: bool
    :param max_trees: Maximum number of trees to construct, defaults to 0.
    :type max_trees: int, optional
    :param max_constructed_variables_per_variable: Maximum number of variables to construct per native variable, defaults to 10.
    :type max_constructed_variables_per_variable: int, optional
    :param result_file: Name of dataset, defaults to "".
    :type result_file: str, optional
    :param results_dir: Path of the results directory, defaults to "Results".
    :type results_dir: str, optional
    :param results_prefixe: Prefix of the result files, defaults to "".
    :type results_prefixe: str, optional
    :param construction_rules: Allowed rules for the automatic variable construction, defaults to kh.all_construction_rules.
    :type construction_rules: list, optional
    :param output_dir: Path of the output directory, defaults to "".
    :type output_dir: str, optional
    """
    def __init__(self,
                 dictionary_file_path,
                 dictionary_name,
                 data_table_path,
                 additional_data_tables,
                 target_variable,
                 exploration_type='Variable',
                 count_effect_reduction=False,
                 max_trees = 0,
                 max_constructed_variables_per_variable = 10,
                 result_file = "",
                 results_dir = "Results",
                 results_prefixe = "",
                 construction_rules = kh.all_construction_rules,
                 output_dir = ""):
        """
        Initialize class AnalyseVariable
        """

        # Dictionary attributes
        self.dictionary_path = dictionary_file_path                             # Khiops dictionary path
        self.dictionary_domain = kh.read_dictionary_file(dictionary_file_path)  # Khiops dictionary domain
        self.dictionary_name = dictionary_name                                  # Name of the dictionary to analyze
        self.root_table_path = data_table_path                                  # Path of the data table 
        self.additional_table = additional_data_tables                          # A dictionary containing the data paths and file paths for a multi-table dictionary file
        self.target = target_variable                                           # Name of the target variable
        self.unused_variable = []                                               # List of unused variable

        # Training attributes
        self.number_agregat = max_constructed_variables_per_variable            # Maximum number of variables to construct per native variable
        self.number_tree = max_trees                                            # Maximum number of trees to construct
        self.result_directory = results_dir                                     # Results directory path
        self.result_prefixe = results_prefixe                                   # Results prefix
        self.construction_rules = construction_rules                            # List of constrcution rules to use
        self.output_dir = output_dir                                            # Output directory path
        self.discretisation = count_effect_reduction                                    # State of discretisation option

        # Other 
        self.exploration_type = exploration_type                                # Exploration type
        self.match_name_dictionary_variable_name = {}                           # Dictionary to match dictionary name with its variable name
        self.match_dictionary_parent_dictionary = {}                            # Dictionary to match dictionary with its parent dictionary
        self.variable_importance = {}                                           # Dictionary to save intermediate variable estimation in case of discretisation
        self.variable_importance_final = []                                     # List of variable importance
        self.variable_importance_group_array = {}                               # Dictionary with pandas dataframe to save variable importance
        self.primitive_importance = self.init_importance_list_primitive()
        self.analyse_count = 0                                                  # Real number of constructed variable
        self.table_exploration_array = pd.DataFrame(                            # Pandas dataframe to save tables information
            [], columns=['Table', 'Type', 'Depth', 'Variable number', 'Categorical variable number', 'Numerical variable number', 'Date variable number'])
        self.variable_exploration_array = pd.DataFrame(                         # Pandas dataframe to save variables information
            [], columns=['Variable', 'Type', 'Table', 'Importance', 'Agregat max.', "Nombre réel d'agrégat"])
        
        # Create directory to save exploration information
        
        exploration_file_name = f"Heuristics_{self.number_agregat}_per_variable_exploration.txt"
        self.exploration_file = open(path.join(output_dir, exploration_file_name), 'w')

    
    def init_importance_list_primitive(self):
        """Initialize the importance primitive list

        :return: Initial list of importance for primitive
        :rtype: list
        """
        importance_primitives = []
        for primitive in self.construction_rules:
            importance_primitives.append([primitive,0])
        return importance_primitives
    
    def match_dictionary_name_variable_name(self):
        """
        Match each secondary dictionary name with its corresponding variable name in the parent dictionary
        and match each secondary dictionary name with its parent dictionary.
        """
        for dictionary in self.dictionary_domain.dictionaries:
            for variable in dictionary.variables:
                if variable.type == 'Table' \
                or variable.type == 'Entity':
                    self.match_name_dictionary_variable_name[variable.name] = variable.object_type
                    self.match_dictionary_parent_dictionary[variable.object_type] = dictionary.name
    
    def get_grouping_variable(self):
        # Initialize python dictionary to match grouping intervals of count variable with tables names
        match_variable_to_add_table_name = {}
        pattern = r'[ ,.;`()]'
        # Train recoder on data
        train_reports_path, modeling_dictionary_path = kh.train_recoder(self.dictionary_path,
                                                                        self.dictionary_name,
                                                                        self.root_table_path,
                                                                        self.target,
                                                                        self.result_directory,
                                                                        additional_data_tables=self.additional_table,
                                                                        results_prefix=self.result_prefixe,
                                                                        informative_variables_only=False,
                                                                        max_constructed_variables=100,
                                                                        max_trees=self.number_tree,
                                                                        construction_rules=self.construction_rules)
        preparation_report = kh.read_analysis_results_file(train_reports_path).preparation_report
        # Add variable to select group
        for agreagat in preparation_report.get_variable_names():
            split_agregat = re.split(pattern, agreagat)
            if (split_agregat[0] == 'Count' and len(split_agregat) == 3) :
                # # Add variable for selection
                IP_count_variable = kh.read_dictionary_file(modeling_dictionary_path).get_dictionary('R_'+self.dictionary_name).get_variable('IdP'+agreagat)
                P_count_variable = kh.read_dictionary_file(modeling_dictionary_path).get_dictionary('R_'+self.dictionary_name).get_variable('P'+agreagat)
                count_variable = kh.read_dictionary_file(modeling_dictionary_path).get_dictionary('R_'+self.dictionary_name).get_variable(agreagat)
                count= kh.Variable()
                count.name = count_variable.name
                count.type = 'Numerical'
                count.used = False
                count.rule = 'Sum('+str(count_variable.rule)+',0)'
                number_group = 1
                if preparation_report.get_variable_statistics(count_variable.name).data_grid != None :
                    number_group = len(preparation_report.get_variable_statistics(count_variable.name).data_grid.dimensions[0].partition)
                match_variable_to_add_table_name[self.match_name_dictionary_variable_name[split_agregat[1]]] = \
                                                                        [IP_count_variable, 
                                                                         P_count_variable, 
                                                                         count, number_group]

        return match_variable_to_add_table_name
    
    def initialize_variable_state(self):
        """
        Initialize dictionary domain with all secondaries variables to unused 
        """
        for dictionary in self.dictionary_domain.dictionaries:
            for variable in dictionary.variables:
                if variable.used == False:
                    self.unused_variable.append(variable.name)
                # Set secondaries table to unused
                if dictionary.root \
                    and (variable.type == 'Table' or variable.type == 'Entity'):
                    self.dictionary_domain.get_dictionary(dictionary.name).get_variable(variable.name).used = False
                elif dictionary.root \
                    and (variable.name != self.target):
                    self.dictionary_domain.get_dictionary(dictionary.name).get_variable(variable.name).used = False
                # Set secondaries variable to unused
                elif not dictionary.root :
                    self.dictionary_domain.get_dictionary(dictionary.name).get_variable(variable.name).used = False
    
    def get_primitives_in_agregat(self, 
                                  derivation_rule):
        """Get the primitives names in an agregat's derivation rule

        :param derivation_rule: Derivation rule 
        :type derivation_rule: str
        :return: List of primitive use in the derivation rule
        :rtype: list
        """
        primitive_list = []     # Init a primitive list
        # Check if a primitive is in the derivation rule
        # 1- Check if the primitive is direcly present in the derivation rule
        for primitive in self.construction_rules:
            if primitive in derivation_rule:
                if primitive not in primitive_list:
                    primitive_list.append(primitive)
        # 2- Check if the primitive is a TableSelection rules and match with its corresponding construction rule
        for primitive in Dict_matching_partition.keys():
            if primitive in derivation_rule:
                if Dict_matching_partition[primitive] not in primitive_list:
                    primitive_list.append(Dict_matching_partition[primitive])
                if "TableSelection" not in primitive_list:
                    primitive_list.append("TableSelection")
        # 3- Check if the primitive name is present in the derivation rule
        # -> primitive name may be present instead of construction rule when
        # multiples primitives are used in the derivation rule  
        for primitive in Dict_matching_rule_name.keys():
            if primitive in derivation_rule:
                if Dict_matching_rule_name[primitive] not in primitive_list:
                    primitive_list.append(Dict_matching_rule_name[primitive])
        return primitive_list

    def update_importance_list_primitive(self, 
                                         primitive_to_update, 
                                         importance):
        """Update the primitive importance list according to new measured importances.

        :param primitive_to_update: List of primitive to be update
        :type primitive_to_update: list
        :param importance: Primitive importance
        :type importance: float
        """
        for i in range(len(self.primitive_importance)) :
            if self.primitive_importance[i][0] in primitive_to_update and self.primitive_importance[i][1]<importance:
                self.primitive_importance[i][1] = importance
        return self.primitive_importance

    def get_importance(self, dictionary, variable, selection_variable="", selection_value=""):
        """Get the importance measure of a variable by univariate analysis.

        :param dictionary: Khiops dictionary where the variable to estimate is.
        :param variable: Variable to estimate.
        :param selection_variable: It trains with only the records such that the value of selection_variable is equal to selection_value, defaults to ""
        :type selection_variable: str, optional
        :param selection_value: See selection_variable option above, defaults to ""
        :type selection_value: str or int or float, optional
        :return: Importance measure
        :rtype: float
        """
        # Initialize importance measure
        variable_importance = 0
        agreagat_max = ""
        label = True
        # Set the variable to estimate to used -> only the variable and its associated table is used 
        self.dictionary_domain.get_dictionary(dictionary.name).get_variable(variable.name).used = True
        # Create variables (agregats)
        train_reports_path, _ = kh.train_recoder(self.dictionary_domain,
                                                                self.dictionary_name,
                                                                self.root_table_path,
                                                                self.target,
                                                                self.result_directory,
                                                                additional_data_tables=self.additional_table,
                                                                results_prefix=variable.name+'_'+selection_value,
                                                                max_constructed_variables=self.number_agregat,
                                                                max_trees=self.number_tree,
                                                                construction_rules=self.construction_rules,
                                                                selection_variable=selection_variable,
                                                                selection_value=selection_value,
                                                                keep_initial_categorical_variables=True,
                                                                keep_initial_numerical_variables=True,
                                                                informative_variables_only=False
                                                                )
        # Update importance measure -> importance measure is the maximum Khiops level in the agregat set.
        preparation_report = kh.read_analysis_results_file(train_reports_path).preparation_report
        for agregat in preparation_report.variables_statistics:
            if agregat.derivation_rule == "TableCount("+get_key(self.match_name_dictionary_variable_name, dictionary.name)+")" \
            and agregat.level != 0:
                label = False
            # Get Varibale importance -> maximum khiops level
            if self.exploration_type == 'Variable' or self.exploration_type == 'All':
                if agregat.level > variable_importance:
                    variable_importance = agregat.level
                    agreagat_max = agregat.name
            # Get Primitive importance -> maximum khiops level
            if self.exploration_type == 'Primitive' or self.exploration_type == 'All':
                primitive_to_update = self.get_primitives_in_agregat(agregat.derivation_rule)
                self.primitive_importance = self.update_importance_list_primitive(primitive_to_update, agregat.level)
                
        # Set the variable estimated to unused
        self.dictionary_domain.get_dictionary(dictionary.name).get_variable(variable.name).used = False
        # Update variable exploration arrays
        if self.variable_exploration_array[(self.variable_exploration_array["Variable"]==variable.name) \
            & (self.variable_exploration_array["Table"]==dictionary.name)].empty:
            row = [variable.name, variable.type, dictionary.name, variable_importance, agreagat_max, len(preparation_report.variables_statistics)]
            self.variable_exploration_array.loc[len(self.variable_exploration_array.index)] = row
        else : 
            if self.variable_exploration_array[self.variable_exploration_array["Variable"]==variable.name].Importance.values[0] < variable_importance:
                self.variable_exploration_array.loc[self.variable_exploration_array["Variable"]==variable.name, 'Importance'] = variable_importance
        self.analyse_count=len(preparation_report.variables_statistics)+self.analyse_count
        return variable_importance, label
    
    def add_variable(self, variable_to_add):
        """Adding khiops variable into dictionary domain

        :param variable_to_add: A list of variables to add into dictionary domain
        :type variable_to_add: list
        """
        for variable in variable_to_add :
            variable.used = False
            self.dictionary_domain.get_dictionary(self.dictionary_name).add_variable(variable)

    def remove_variable(self, variable_to_remove):
        """Remove khiops variable from dictionary domain

        :param variable_to_remove: A list of variables to remove from dictionary domain
        :type variable_to_remove: list
        """
        for variable in variable_to_remove :
            self.dictionary_domain.get_dictionary(self.dictionary_name).remove_variable(variable.name)

    def get_variable_number(self, variable_to_add):
        """Get the number of variable in dataset

        :param variable_to_add: A list of variables to add into dictionary domain
        :type variable_to_add: list
        :return: Number of variable 
        :rtype: int
        """
        variable_number = 0
        for dictionary in self.dictionary_domain.dictionaries:
            if not dictionary.root:
                if dictionary.name in variable_to_add.keys():
                    variable_number += len(dictionary.variables)*variable_to_add[dictionary.name][-1]
                else :
                    variable_number += len(dictionary.variables)
        return variable_number
    
    def write_exploration_file(self):
        """Create a text file with exploration and estimation information
        """
        self.exploration_file.write("==========================================================================================================\n")
        self.exploration_file.write("Variable and/or primitive exploration information\n")
        self.exploration_file.write("==========================================================================================================\n")
        self.exploration_file.write(f"Agregats number per variable : \t {self.number_agregat} \n")
        self.exploration_file.write(f"Real analyse agregats number : \t {self.analyse_count} \n")
        self.exploration_file.write(f"Discretisation : \t {self.discretisation} \n")
        self.exploration_file.write(f"Exploration for : \t {self.exploration_type} \n")
        self.exploration_file.write("\n")
        self.exploration_file.write("==========================================================================================================\n")
        self.exploration_file.write(self.variable_exploration_array.to_markdown())
        self.exploration_file.write("\n")
        self.exploration_file.write("\n")
        self.exploration_file.write("==========================================================================================================\n")
        self.exploration_file.write("Table exploration information\n")
        self.exploration_file.write("==========================================================================================================\n")
        self.exploration_file.write(self.table_exploration_array.to_markdown())
        self.exploration_file.write("\n")
        self.exploration_file.write("\n")
        self.exploration_file.write("\n")
        if self.discretisation:
            self.exploration_file.write("================================================================================================================================================================================\n")
            self.exploration_file.write("================================================================================================================================================================================\n")
            self.exploration_file.write("Importance measure for variables depending of group and table\n")
            self.exploration_file.write("================================================================================================================================================================================\n")
            for key in self.variable_importance.keys():
                self.exploration_file.write(f"-------Table------- {key} \n")
                if self.exploration_type=='All' or self.exploration_type=='Variable':
                    self.exploration_file.write(self.variable_importance_group_array[key].to_markdown())
                    self.exploration_file.write("\n")
                    self.exploration_file.write("\n")
            self.exploration_file.write("\n")
            self.exploration_file.write("\n")
        if self.exploration_type=='All' or self.exploration_type=='Variable':
            self.exploration_file.write("================================================================================================================================================================================\n")
            self.exploration_file.write("Final importance measure for variables \n")
            for item in self.variable_importance_final:
                self.exploration_file.write(str(item))
                self.exploration_file.write("\n")
        if self.exploration_type=='All' or self.exploration_type=='Primitive':
            self.exploration_file.write("================================================================================================================================================================================\n")
            self.exploration_file.write("Final importance measure for primitive \n")
            for item in self.primitive_importance:
                self.exploration_file.write(str(item))
                self.exploration_file.write("\n")
        self.exploration_file.write("================================================================================================================================================================================\n")
        self.exploration_file.write("================================================================================END=====================================================================================================================================================\n")
        self.exploration_file.close()

    def univariate_analyse_by_discretisation(self):
        """
        Analyse variables by estimated a measure of importance for each variables using a 
        n univariate analysis with discretisation.
        Create a sorted list of importance.
        """
        # Get count grouping variable to add into dictionary domain -> depending of the table
        variable_to_add = self.get_grouping_variable()
        # Analyse by table
        for dictionary in tqdm(self.dictionary_domain.dictionaries):
            if not dictionary.root:
                # Set the table to analyse to used
                self.variable_importance_group_array[dictionary.name] = pd.DataFrame()
                self.variable_importance[dictionary.name] = {}
                self.dictionary_domain.get_dictionary(self.match_dictionary_parent_dictionary[dictionary.name]).\
                    get_variable(get_key(self.match_name_dictionary_variable_name,dictionary.name)).used = True
                # Add variable for the selection -> only for Table type table
                if dictionary.name in variable_to_add.keys():
                    self.add_variable(variable_to_add[dictionary.name][:-1])
                    # Number of group to split instances
                    number_group = variable_to_add[dictionary.name][-1]
                    for i in range(number_group):
                        list_importance = []
                        categorical_variable = 0
                        numerical_variable = 0
                        date_variable = 0
                        label = True
                        # Get importance measure for each variable of the table
                        for variable in tqdm(dictionary.variables):
                            # Get the number of variable type in tables
                            if variable.type == "Categorical":
                                categorical_variable+=1
                            elif variable.type == "Numerical":
                                numerical_variable+=1
                            else :
                                date_variable+=1
                            if variable.name not in self.unused_variable : 
                                variable_importance, keep = self.get_importance(dictionary, 
                                                                        variable,
                                                                        selection_variable=variable_to_add[dictionary.name][0].name,
                                                                        selection_value='I'+str(i+1))
                                list_importance.append((dictionary.name, variable.name,variable_importance))
                                if keep == False:
                                    label = False
                        # Store importance list by table and by group
                        self.variable_importance_group_array[dictionary.name][dictionary.name+'/'+'I'+str(i+1)] = sorted(list_importance, key=lambda level: level[2])
                        self.variable_importance[dictionary.name]['I'+str(i+1)] = sorted(list_importance, key=lambda level: level[2])
                        self.variable_importance[dictionary.name]['I'+str(i+1)].append(label)
                    # Remove variable selection of the table
                    self.remove_variable(variable_to_add[dictionary.name][:-1])
                # get unique importance list of Entity type table
                else : 
                    categorical_variable = 0
                    numerical_variable = 0
                    date_variable = 0
                    list_importance = []
                    label = True
                    for variable in tqdm(dictionary.variables):
                        # Get the number of variable type in tables
                        if variable.type == "Categorical":
                            categorical_variable+=1
                        elif variable.type == "Numerical":
                            numerical_variable+=1
                        else :
                            date_variable+=1
                        if variable.name not in self.unused_variable : 
                            variable_importance, keep = self.get_importance(dictionary, variable)
                            list_importance.append((dictionary.name, variable.name, variable_importance))
                    # Store importance list by table and by group
                    self.variable_importance_group_array[dictionary.name][dictionary.name] = sorted(list_importance, key=lambda level: level[2])
                    self.variable_importance[dictionary.name]['1 group'] = sorted(list_importance, key=lambda level: level[2])
                    self.variable_importance[dictionary.name]['1 group'].append(label)
                # Set the table to analyse to unused
                self.dictionary_domain.get_dictionary(self.match_dictionary_parent_dictionary[dictionary.name]).\
                        get_variable(get_key(self.match_name_dictionary_variable_name,dictionary.name)).used = False
            else :
                categorical_variable = 0
                numerical_variable = 0
                date_variable = 0
                list_importance = []
                for variable in tqdm(dictionary.variables):
                    # Get the number of variable type in tables
                    if variable.type == "Categorical":
                        categorical_variable+=1
                    elif variable.type == "Numerical":
                        numerical_variable+=1
                    else :
                        date_variable+=1
            self.table_exploration_array.loc[len(self.variable_exploration_array.index)] = [dictionary.name, None, 0, len(dictionary.variables), categorical_variable, numerical_variable, date_variable]  
            print("self.table_exploration_array")
            print(self.table_exploration_array)

        # Update the final variable list of importance -> combine importance list of each variable
        self.variable_exploration_array.to_csv(self.output_dir + "/exploration_array.csv")

    def univariate_analyse(self):
        """Analyse variables by estimated a measure of importance for each variables using an
        univariate analysis.
        Create a sorted list of importance.
        """
        for dictionary in tqdm(self.dictionary_domain.dictionaries):
            if not dictionary.root:
                # Set the table to analyse to used
                self.dictionary_domain.get_dictionary(self.match_dictionary_parent_dictionary[dictionary.name]).\
                    get_variable(get_key(self.match_name_dictionary_variable_name,dictionary.name)).used = True
                categorical_variable = 0
                numerical_variable = 0
                date_variable = 0
                # Get importance measure for each variable of the table
                for variable in tqdm(dictionary.variables):
                    # Get the number of variable type in tables
                    if variable.type == "Categorical":
                        categorical_variable+=1
                    elif variable.type == "Numerical":
                        numerical_variable+=1
                    else :
                        date_variable+=1
                    if variable.name not in self.unused_variable : 
                        # Get variable and/or primitive importance
                        variable_importance, _ = self.get_importance(dictionary, 
                                                                variable)
                        self.variable_importance_final.append((dictionary.name, variable.name,variable_importance))
                        
        self.variable_importance_final = sorted(self.variable_importance_final, key=lambda level: level[2])
        print("self.variable_importance_final")
        print(self.variable_importance_final)

    def in_list(self, list, dictionary, value):
        """Return value index in a python list

        :param list: Python list
        :type list: list
        :param dictionary: Dictionary name where the variable is present
        :type dictionary: str
        :param value: Value to locate in list.
        :type value: str
        :return: Index of the value in the list if existing.
        :rtype: int
        """
        for i in range(len(list)):
            if list[i][0] == dictionary and list[i][1] == value:
                return i
        return -1
    
    def get_final_importance_list(self):
        """
        Get the final variable importance list.
        """
        # Init the final importance list
        list_importance = []
        for key, value in self.variable_importance.items():
            for key2, value2 in value.items():
                label = value2.pop()
                if label == False:
                    continue
                for dictionary, variable, level in value2:
                    index_variable = self.in_list(list_importance, dictionary, variable)
                    if index_variable != -1 \
                        and level > list_importance[index_variable][2]:
                        list_importance[index_variable] = (dictionary, variable, level)
                    elif index_variable == -1:
                        list_importance.append((dictionary, variable, level))
        self.variable_importance_final += sorted(list_importance, key=lambda level: level[2])
                        
    def show_importance_list(self):
        """Print variable importance list
        """
        for key, value in self.variable_importance.items():
            print(f"----------------Table {key}--------------------")
            for key2, value2 in value.items():
                print(f"----Group {key2}---")
                print(value2)

    def analyse_variable(self):
        """Global function to estimate variable's importance
        """
        # Get matching dictionary for variable table name and table name
        self.match_dictionary_name_variable_name()
        # Initialise dictionary domain -> all variable to unused
        self.initialize_variable_state()
        
        if self.discretisation:
            self.univariate_analyse_by_discretisation()
        else:
            self.univariate_analyse()
        # Sorted importances lists
        self.variable_importance_final = sorted(self.variable_importance_final, key=lambda level: level[2])
        self.primitive_importance = sorted(self.primitive_importance, key=lambda level: level[1])
        self.write_exploration_file()

    
