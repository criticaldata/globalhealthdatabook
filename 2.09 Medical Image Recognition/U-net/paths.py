import os
import inspect

def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

current_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(current_filename))
input_data_dir = os.path.join(root_dir, 'input_data')
response_output_dir = os.path.join(root_dir, 'predicted_response')

dir_to_make = [input_data_dir, response_output_dir]
mkdir_if_not_exist(dir_list= dir_to_make)
